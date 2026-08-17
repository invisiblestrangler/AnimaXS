#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

/// Runtime-only bridge for the A12/H11 Apple Neural Engine path proven by the
/// AnimaANEProbe v14 device harness. No private Apple headers are imported;
/// the private runtime is resolved dynamically and this backend remains an
/// opt-in, device-only experimental path.
FOUNDATION_EXPORT BOOL A12ANEIsAvailable(void);
FOUNDATION_EXPORT NSString *A12ANERuntimeStatus(void);

@interface A12ANESurface : NSObject
@property(nonatomic, readonly) id<MTLBuffer> metalBuffer;
@property(nonatomic, readonly) NSUInteger channels;
@property(nonatomic, readonly) NSUInteger spatial;
@property(nonatomic, readonly) NSUInteger planeStrideElements;
@property(nonatomic, readonly) NSUInteger byteCount;
- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                               channels:(NSUInteger)channels
                                spatial:(NSUInteger)spatial
                                  error:(NSError **)error;
@end

@interface A12ANEProjectionModel : NSObject
@property(nonatomic, readonly) NSUInteger inputChannels;
@property(nonatomic, readonly) NSUInteger outputChannels;
@property(nonatomic, readonly) NSUInteger spatial;
@property(nonatomic, readonly) double loadMilliseconds;
@property(nonatomic, readonly) NSString *label;

/// qBytes is row-major U8 [outputChannels,inputChannels]. bias/scale are one
/// Float32 value per output channel, matching the H11 Espresso W8 ABI.
- (nullable instancetype)initWithQBytes:(NSData *)qBytes
                                biasF32:(NSData *)biasF32
                               scaleF32:(NSData *)scaleF32
                          inputChannels:(NSUInteger)inputChannels
                         outputChannels:(NSUInteger)outputChannels
                                spatial:(NSUInteger)spatial
                                  label:(NSString *)label
                               cacheKey:(NSString *)cacheKey
                                  error:(NSError **)error;

- (BOOL)evaluateInput:(A12ANESurface *)input
                output:(A12ANESurface *)output
           milliseconds:(nullable double *)milliseconds
                  error:(NSError **)error;
- (void)invalidate;
@end

@interface A12ANEQKVModel : NSObject
@property(nonatomic, readonly) NSUInteger channels;
@property(nonatomic, readonly) NSUInteger spatial;
@property(nonatomic, readonly) double loadMilliseconds;
@property(nonatomic, readonly) NSString *label;

/// Fused self-attention Q/K/V projection. All three W8 matrices are
/// row-major U8 [channels,channels] with one Float32 bias/scale per row.
- (nullable instancetype)initWithQBytes:(NSData *)qBytes
                                qBiasF32:(NSData *)qBiasF32
                               qScaleF32:(NSData *)qScaleF32
                                  kBytes:(NSData *)kBytes
                                kBiasF32:(NSData *)kBiasF32
                               kScaleF32:(NSData *)kScaleF32
                                  vBytes:(NSData *)vBytes
                                vBiasF32:(NSData *)vBiasF32
                               vScaleF32:(NSData *)vScaleF32
                                channels:(NSUInteger)channels
                                 spatial:(NSUInteger)spatial
                                   label:(NSString *)label
                                cacheKey:(NSString *)cacheKey
                                   error:(NSError **)error;

- (BOOL)evaluateInput:(A12ANESurface *)input
               qOutput:(A12ANESurface *)qOutput
               kOutput:(A12ANESurface *)kOutput
               vOutput:(A12ANESurface *)vOutput
          milliseconds:(nullable double *)milliseconds
                 error:(NSError **)error;
- (void)invalidate;
@end

NS_ASSUME_NONNULL_END
