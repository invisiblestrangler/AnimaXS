import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Errors thrown by the ANMA parser and model validation.
///
/// `LocalizedError` conformance ensures `error.localizedDescription` surfaces
/// the actual reason (e.g. "model SHA-256 does not match manifest") instead of
/// a generic "The operation couldn't be completed. (AnimaXS.AnimapkError
/// error 3.)" message.
enum AnimapkError: Error, CustomStringConvertible, LocalizedError {
    case io(String)
    case header(String)
    case json(String)
    case validation(String)

    var description: String {
        switch self {
        case .io(let m), .header(let m), .json(let m), .validation(let m): return m
        }
    }

    var errorDescription: String? {
        description
    }
}

/// Read-only POSIX mmap wrapper. Owns the fd and mapping; unmaps/closes on deinit.
final class MappedFile {
    let fileSize: Int
    private let fd: Int32
    private let base: UnsafeMutableRawPointer

    init(url: URL) throws {
        let path = url.path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw AnimapkError.io("open failed for \(path): \(String(cString: strerror(errno)))") }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            close(fd)
            throw AnimapkError.io("fstat failed for \(path)")
        }
        let size = Int(st.st_size)
        guard size > 0 else {
            close(fd)
            throw AnimapkError.io("file is empty: \(path)")
        }
        guard let base = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0),
              base != UnsafeMutableRawPointer(bitPattern: -1) else {
            close(fd)
            throw AnimapkError.io("mmap failed for \(path)")
        }
        self.fd = fd
        self.base = base
        self.fileSize = size
    }

    /// Entire mapped range.
    func bytes() -> UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(start: base, count: fileSize)
    }

    /// Pointer at a byte offset.
    func pointer(offset: Int) -> UnsafeRawPointer {
        UnsafeRawPointer(base.advanced(by: offset))
    }

    deinit {
        munmap(base, fileSize)
        close(fd)
    }
}
