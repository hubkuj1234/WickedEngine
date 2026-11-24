#pragma once
#include "wiMath.h"

#include <string>

// The purpose of this helper is to expose functionality that is specific to Apple platform and must be implemented in Objective C++

namespace wi::apple
{
void SetMetalLayerToWindow(void* window, void* layer);
void* GetViewFromWindow(void* window);
XMUINT2 GetWindowSizeNoScaling(void* handle);
XMUINT2 GetWindowSize(void* handle);
float GetDPIForWindow(void* handle);
XMFLOAT2 GetMousePositionInWindow(void* handle);
void SetMousePositionInWindow(void* window, XMFLOAT2 value);
int MessageBox(const char* title, const char* message, const char* buttons = nullptr);
std::string GetResourcePath();
std::string GetExecutablePath();
std::string GetApplicationSupportPath();
void CursorInit(void** cursor_table);
void CursorSet(void* cursor);
void CursorHide(bool hide);
void SetWindowFullScreen(void* handle, bool fullscreen);
bool IsWindowFullScreen(void* handle);
void OpenUrl(const char* url);
std::string GetClipboardText();
void SetClipboardText(const char* str);
void SetClipboardImagePNG(const void* data, size_t size);
void* CreateCursorFromARGB8ImageData(const void* data, uint32_t width, uint32_t height, int hotspotX, int hotspotY);
void GetSafeArea(void* window, float& left, float& right, float& top, float& bottom);
void* CreateMetalDisplayLink(void* ca_metal_layer, double preferred_refresh_hz);
void DestroyMetalDisplayLink(void* link_handle);
void SetMetalDisplayLinkRefreshRate(void* link_handle, double refresh_hz);
bool WaitForMetalDisplayLinkOpportunity(void* link_handle, double timeout_seconds, void** out_retained_drawable, double* out_target_present_sec, double* out_refresh_interval_sec, bool* out_vrr_active, bool* out_timing_measured);
bool WindowSupportsHDR(void* ns_window_ptr);
void ConfigureMetalLayerHDR(void* ca_metal_layer, void* ns_window_ptr, bool allow_hdr, bool* out_hdr_active, bool* out_hdr_supported, float* out_max_edr);
void AttachMetalLayerResidencySet(void* ca_metal_layer, void* mtl4_queue);
void CommitMetal4CommandBuffers(void* mtl4_queue, void* const* cmd_buffers, size_t count, const char* queue_label);
}
