#ifdef __APPLE__

#include "wiAppleHelper.h"

#import <AppKit/AppKit.h>
#import <Cocoa/Cocoa.h>
#import <CoreFoundation/CoreFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <Metal/MTL4CommandQueue.h>
#import <QuartzCore/CAMetalDisplayLink.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace wi::apple {
struct MetalDisplayLinkState;
void notify_metal_display_link(void* state_ptr, void* update_ptr);
} // namespace wi::apple

@interface DaisyWiMetalDisplayLinkTarget : NSObject <CAMetalDisplayLinkDelegate>
@property(nonatomic, assign) void* state;
@end

@implementation DaisyWiMetalDisplayLinkTarget
- (void)metalDisplayLink:(CAMetalDisplayLink*)link
             needsUpdate:(CAMetalDisplayLinkUpdate*)update {
    (void)link;
    wi::apple::notify_metal_display_link(self.state, (__bridge void*)update);
}
@end

namespace wi::apple {

namespace {

double steady_now_seconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace

struct MetalDisplayLinkState {
    CAMetalLayer* layer = nil;
    DaisyWiMetalDisplayLinkTarget* target = nil;
    CAMetalDisplayLink* link = nil;
    NSThread* thread = nil;
    CFRunLoopRef run_loop = nullptr;
    std::atomic<bool> stop{false};
    std::atomic<double> preferred_refresh_hz{120.0};
    bool uses_display_link = false;

    std::mutex mutex;
    std::condition_variable cv;
    id<CAMetalDrawable> pending_drawable = nil;
    double pending_target_present_steady = 0.0;
    double pending_refresh_interval = 1.0 / 60.0;
    double last_target_media_time = 0.0;
    bool has_tick = false;
    bool vrr_active = false;

    void on_update(CAMetalDisplayLinkUpdate* update) {
        const CFTimeInterval media_now = CACurrentMediaTime();
        const double steady_now = steady_now_seconds();
        const CFTimeInterval target_ts_media = update.targetTimestamp;
        const CFTimeInterval target_present_media =
            update.targetPresentationTimestamp;
        std::lock_guard<std::mutex> lock(mutex);
        if (last_target_media_time > 0.0) {
            const double observed_dt =
                static_cast<double>(target_ts_media - last_target_media_time);
            if (observed_dt > 0.001 && observed_dt < 0.2) {
                pending_refresh_interval =
                    pending_refresh_interval * 0.8 + observed_dt * 0.2;
            }
        } else {
            const double hz =
                std::max(preferred_refresh_hz.load(std::memory_order_acquire), 1.0);
            pending_refresh_interval = 1.0 / hz;
        }
        // Always aim one refresh interval ahead on the display-link commit grid
        // instead of flipping between targetTimestamp (~0 ms) and
        // targetPresentationTimestamp (~2-3 refreshes ahead).
        const double delta_to_present =
            (target_ts_media > media_now + 0.0005)
                ? static_cast<double>(target_ts_media - media_now)
                : pending_refresh_interval;
        const double target_steady = steady_now + delta_to_present;
        last_target_media_time = static_cast<double>(target_ts_media);
        pending_drawable = update.drawable;
        pending_target_present_steady = target_steady;
        has_tick = true;
        cv.notify_all();
    }

    void start() {
        if (!layer) {
            return;
        }
        if (@available(macOS 14.0, *)) {
            stop.store(false, std::memory_order_release);
            target = [[DaisyWiMetalDisplayLinkTarget alloc] init];
            target.state = this;
            MetalDisplayLinkState* self = this;
            thread = [[NSThread alloc] initWithBlock:^{
                @autoreleasepool {
                    CAMetalDisplayLink* dl =
                        [[CAMetalDisplayLink alloc] initWithMetalLayer:self->layer];
                    dl.delegate = self->target;
                    dl.preferredFrameLatency = 1.0f;
                    const float max_hz = static_cast<float>(std::clamp(
                        self->preferred_refresh_hz.load(std::memory_order_acquire),
                        24.0, 240.0));
                    // Allow ProMotion / VRR to adapt down smoothly if UI cannot
                    // maintain max refresh, while targeting full max refresh.
                    const float min_hz = std::max(24.0f, max_hz * 0.25f);
                    dl.preferredFrameRateRange =
                        CAFrameRateRangeMake(min_hz, max_hz, max_hz);
                    self->link = dl;
                    self->vrr_active = (max_hz > 60.5f);

                    NSRunLoop* loop = [NSRunLoop currentRunLoop];
                    self->run_loop = [loop getCFRunLoop];
                    [dl addToRunLoop:loop forMode:NSDefaultRunLoopMode];
                    while (!self->stop.load(std::memory_order_acquire)) {
                        @autoreleasepool {
                            [loop runMode:NSDefaultRunLoopMode
                               beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.05]];
                        }
                    }
                    [dl invalidate];
                    if (self->link == dl) {
                        self->link = nil;
                    }
                    self->run_loop = nullptr;
                }
            }];
            thread.name = @"DaisyWiMetalDisplayLink";
            [thread start];
            uses_display_link = true;
        }
    }

    void shutdown() {
        if (target) {
            target.state = nullptr;
        }
        stop.store(true, std::memory_order_release);
        cv.notify_all();
        if (@available(macOS 14.0, *)) {
            if (link) {
                link.paused = YES;
            }
        }
        if (run_loop) {
            CFRunLoopWakeUp(run_loop);
        }
        NSThread* t = thread;
        thread = nil;
        if (t) {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!t.isFinished && std::chrono::steady_clock::now() < deadline) {
                [NSThread sleepForTimeInterval:0.001];
            }
        }
        link = nil;
        run_loop = nullptr;
        target = nil;
        uses_display_link = false;
        std::lock_guard<std::mutex> lock(mutex);
        pending_drawable = nil;
        has_tick = false;
    }
};

void notify_metal_display_link(void* state_ptr, void* update_ptr) {
    if (!state_ptr || !update_ptr) {
        return;
    }
    auto* state = static_cast<MetalDisplayLinkState*>(state_ptr);
    state->on_update((__bridge CAMetalDisplayLinkUpdate*)update_ptr);
}

void* CreateMetalDisplayLink(void* ca_metal_layer, double preferred_refresh_hz) {
    if (!ca_metal_layer) {
        return nullptr;
    }
    auto* state = new MetalDisplayLinkState();
    state->layer = (__bridge CAMetalLayer*)ca_metal_layer;
    state->preferred_refresh_hz.store(
        std::clamp(preferred_refresh_hz, 24.0, 240.0),
        std::memory_order_release);
    state->pending_refresh_interval =
        1.0 / std::clamp(preferred_refresh_hz, 24.0, 240.0);
    state->start();
    return state;
}

void DestroyMetalDisplayLink(void* link_handle) {
    if (!link_handle) {
        return;
    }
    auto* state = static_cast<MetalDisplayLinkState*>(link_handle);
    state->shutdown();
    delete state;
}

void SetMetalDisplayLinkRefreshRate(void* link_handle, double refresh_hz) {
    if (!link_handle) {
        return;
    }
    auto* state = static_cast<MetalDisplayLinkState*>(link_handle);
    const double clamped = std::clamp(refresh_hz, 24.0, 240.0);
    const double prev = state->preferred_refresh_hz.exchange(
        clamped, std::memory_order_acq_rel);
    if (std::abs(prev - clamped) < 0.5) {
        return;
    }
    if (@available(macOS 14.0, *)) {
        if (state->link) {
            const float max_hz = static_cast<float>(clamped);
            const float min_hz = std::max(24.0f, max_hz * 0.25f);
            state->link.preferredFrameRateRange =
                CAFrameRateRangeMake(min_hz, max_hz, max_hz);
            state->vrr_active = (max_hz > 60.5f);
        }
    }
}

bool WaitForMetalDisplayLinkOpportunity(void* link_handle,
                                        double timeout_seconds,
                                        void** out_retained_drawable,
                                        double* out_target_present_sec,
                                        double* out_refresh_interval_sec,
                                        bool* out_vrr_active,
                                        bool* out_timing_measured) {
    if (out_retained_drawable) *out_retained_drawable = nullptr;
    if (out_target_present_sec) *out_target_present_sec = 0.0;
    if (out_refresh_interval_sec) *out_refresh_interval_sec = 1.0 / 60.0;
    if (out_vrr_active) *out_vrr_active = false;
    if (out_timing_measured) *out_timing_measured = false;

    if (!link_handle) {
        return false;
    }
    auto* state = static_cast<MetalDisplayLinkState*>(link_handle);
    if (!state->uses_display_link) {
        return false;
    }

    std::unique_lock<std::mutex> lock(state->mutex);
    const auto timeout = std::chrono::duration<double>(
        std::clamp(timeout_seconds, 0.001, 0.5));
    const bool ok = state->cv.wait_for(lock, timeout, [state]() {
        return (state->has_tick && state->pending_drawable != nil) ||
               state->stop.load(std::memory_order_acquire);
    });
    if (!ok || !state->has_tick || state->pending_drawable == nil) {
        return false;
    }

    id<CAMetalDrawable> drawable = state->pending_drawable;
    state->pending_drawable = nil;
    state->has_tick = false;
    if (out_retained_drawable) {
        // Caller (metal-cpp NS::TransferPtr) takes ownership of +1 retain.
        *out_retained_drawable = (__bridge_retained void*)drawable;
    }
    if (out_target_present_sec) {
        *out_target_present_sec = state->pending_target_present_steady;
    }
    if (out_refresh_interval_sec) {
        *out_refresh_interval_sec = state->pending_refresh_interval;
    }
    if (out_vrr_active) {
        *out_vrr_active = state->vrr_active;
    }
    if (out_timing_measured) {
        *out_timing_measured = true;
    }
    return true;
}

bool WindowSupportsHDR(void* ns_window_ptr) {
    if (!ns_window_ptr) {
        return false;
    }
    NSWindow* window = (__bridge NSWindow*)ns_window_ptr;
    NSScreen* screen = window.screen ? window.screen : [NSScreen mainScreen];
    if (!screen) {
        return false;
    }
    if (@available(macOS 10.15, *)) {
        return screen.maximumPotentialExtendedDynamicRangeColorComponentValue > 1.05;
    }
    return false;
}

void ConfigureMetalLayerHDR(void* ca_metal_layer,
                            void* ns_window_ptr,
                            bool allow_hdr,
                            bool* out_hdr_active,
                            bool* out_hdr_supported,
                            float* out_max_edr) {
    bool supported = WindowSupportsHDR(ns_window_ptr);
    float max_edr = 1.0f;
    if (ns_window_ptr) {
        NSWindow* window = (__bridge NSWindow*)ns_window_ptr;
        NSScreen* screen = window.screen ? window.screen : [NSScreen mainScreen];
        if (screen) {
            if (@available(macOS 10.15, *)) {
                max_edr = static_cast<float>(
                    std::max<CGFloat>(1.0, screen.maximumExtendedDynamicRangeColorComponentValue));
            }
        }
    }
    bool active = allow_hdr && supported;
    if (ca_metal_layer) {
        CAMetalLayer* layer = (__bridge CAMetalLayer*)ca_metal_layer;
        layer.presentsWithTransaction = NO;
        if (@available(macOS 10.15, *)) {
            layer.wantsExtendedDynamicRangeContent = active ? YES : NO;
        }
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(
            active ? kCGColorSpaceExtendedLinearSRGB : kCGColorSpaceSRGB);
        if (cs) {
            layer.colorspace = cs;
            CGColorSpaceRelease(cs);
        }
    }
    if (out_hdr_active) *out_hdr_active = active;
    if (out_hdr_supported) *out_hdr_supported = supported;
    if (out_max_edr) *out_max_edr = max_edr;
}

void AttachMetalLayerResidencySet(void* ca_metal_layer, void* mtl4_queue) {
    if (!ca_metal_layer || !mtl4_queue) {
        return;
    }
    if (@available(macOS 26.0, *)) {
        CAMetalLayer* layer = (__bridge CAMetalLayer*)ca_metal_layer;
        id<MTL4CommandQueue> queue = (__bridge id<MTL4CommandQueue>)mtl4_queue;
        id<MTLResidencySet> rset = layer.residencySet;
        if (rset != nil) {
            [queue addResidencySet:rset];
        }
    }
}

void CommitMetal4CommandBuffers(void* mtl4_queue,
                                void* const* cmd_buffers,
                                size_t count,
                                const char* queue_label) {
    if (!mtl4_queue || !cmd_buffers || count == 0) {
        return;
    }
    if (@available(macOS 26.0, *)) {
        @autoreleasepool {
            id<MTL4CommandQueue> queue = (__bridge id<MTL4CommandQueue>)mtl4_queue;
            const char* label = queue_label ? queue_label : "?";
            static std::atomic<uint64_t> s_commit_seq{0};
            const uint64_t seq = s_commit_seq.fetch_add(1, std::memory_order_relaxed);
            auto* cbs = reinterpret_cast<const id<MTL4CommandBuffer> __unsafe_unretained*>(cmd_buffers);
            MTL4CommitOptions* opts = [[MTL4CommitOptions alloc] init];
            [opts addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
                NSError* err = feedback.error;
                if (err != nil) {
                    std::fprintf(
                        stderr,
                        "[Metal4 CommitFeedback ERROR] queue=%s seq=%llu count=%zu gpu=%.1fms domain=%s code=%ld desc=%s\n",
                        label,
                        static_cast<unsigned long long>(seq),
                        count,
                        (feedback.GPUEndTime - feedback.GPUStartTime) * 1000.0,
                        err.domain ? err.domain.UTF8String : "(null)",
                        static_cast<long>(err.code),
                        err.localizedDescription ? err.localizedDescription.UTF8String : "(null)");
                    std::fflush(stderr);
                }
            }];
            [queue commit:cbs count:static_cast<NSUInteger>(count) options:opts];
        }
    }
}

} // namespace wi::apple

#endif // __APPLE__
