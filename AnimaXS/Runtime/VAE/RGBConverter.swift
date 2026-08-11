import Foundation
import UIKit

enum RGBConverter {
    /// Converts source CHW RGB in approximately [-1,1] to interleaved RGBA8.
    static func rgba8(_ rgb: [Float], width: Int, height: Int) throws -> [UInt8] {
        let pixels = width * height
        guard width > 0, height > 0, rgb.count == pixels * 3 else {
            throw AnimapkError.validation("invalid decoded RGB buffer")
        }
        var output = [UInt8](repeating: 255, count: pixels * 4)
        for pixel in 0..<pixels {
            for channel in 0..<3 {
                let normalized = min(max((rgb[channel * pixels + pixel] + 1) * 0.5, 0), 1)
                output[pixel * 4 + channel] = UInt8((normalized * 255).rounded())
            }
        }
        return output
    }

    static func image(_ rgb: [Float], width: Int, height: Int) throws -> UIImage {
        let rgba = try rgba8(rgb, width: width, height: height)
        let data = Data(rgba) as CFData
        guard let provider = CGDataProvider(data: data),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let cgImage = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent) else {
            throw AnimapkError.validation("failed to create decoded RGB image")
        }
        return UIImage(cgImage: cgImage)
    }
}
