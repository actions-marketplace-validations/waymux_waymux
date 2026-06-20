/*
 * waymux ffv1_vulkan shim.
 *
 * ffmpeg-sys-next 8.1 doesn't bind libavutil/hwcontext_vulkan.h, so the
 * Vulkan-specific hwcontext fields (AVVulkanDeviceContext, AVVulkanFramesContext,
 * AVVkFrame) are not visible from Rust. Rather than hand-mirror those
 * structs (which embed VkPhysicalDeviceFeatures2 and other large/volatile
 * types) or pull bindgen into our build, we expose a tiny C surface that
 * pokes the fields we need.
 *
 * All functions return 0 on success and a negative AVERROR-style code on
 * failure. Pointers are passed as opaque (void *) on the Rust side.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/version.h>
#include <vulkan/vulkan.h>

/*
 * AVVulkanDeviceContext gained the qf[]/nb_qf queue-family array in ffmpeg 7
 * (libavutil 59); ffmpeg 6 (libavutil 58, e.g. Ubuntu 24.04 LTS) only has the
 * fixed queue_family_*_index fields. Gate on the major version so the shim
 * compiles on both.
 */
#if LIBAVUTIL_VERSION_MAJOR >= 59
#define WAYMUX_AVVK_HAVE_QF_ARRAY 1
#else
#define WAYMUX_AVVK_HAVE_QF_ARRAY 0
#endif

/*
 * Configure an allocated-but-not-yet-init'd AVHWDeviceContext (VULKAN) to
 * wrap waymux's existing VkInstance / VkPhysicalDevice / VkDevice.
 *
 * Must be called BEFORE av_hwdevice_ctx_init(hw_device_ref).
 *
 * compute_qf / encode_qf: queue family indices.
 * encode_qf may be UINT32_MAX to signal "no video encode queue".
 */
int waymux_avvk_set_device(
    AVBufferRef *hw_device_ref,
    void *instance,
    void *phys_dev,
    void *act_dev,
    void *get_proc_addr,
    uint32_t compute_qf,
    uint32_t encode_qf,
    const char * const *enabled_dev_extensions,
    int nb_enabled_dev_extensions)
{
    if (!hw_device_ref) return -1;
    AVHWDeviceContext *dev_ctx = (AVHWDeviceContext *)hw_device_ref->data;
    if (!dev_ctx || dev_ctx->type != AV_HWDEVICE_TYPE_VULKAN) return -2;
    AVVulkanDeviceContext *vk = (AVVulkanDeviceContext *)dev_ctx->hwctx;
    if (!vk) return -3;

    vk->alloc = NULL;
    vk->get_proc_addr = (PFN_vkGetInstanceProcAddr)get_proc_addr;
    vk->inst = (VkInstance)instance;
    vk->phys_dev = (VkPhysicalDevice)phys_dev;
    vk->act_dev = (VkDevice)act_dev;

    /*
     * device_features is filled in by av_hwdevice_ctx_init() based on
     * what the physical device supports; leaving it zeroed is fine because
     * we already enabled the features we need when we created act_dev.
     */
    memset(&vk->device_features, 0, sizeof(vk->device_features));

    vk->enabled_inst_extensions = NULL;
    vk->nb_enabled_inst_extensions = 0;
    /*
     * libav's hevc_vulkan / h264_vulkan encoders gate on the enabled
     * device extensions list — they refuse to open if their codec's
     * extension isn't present here. Callers MUST pass the list they
     * enabled at vkCreateDevice time (VK_KHR_video_encode_queue,
     * VK_KHR_video_encode_h264, VK_KHR_video_encode_h265, etc.).
     * Pointers must remain valid for the lifetime of the AVVulkan-
     * DeviceContext — we just stash them through.
     */
    vk->enabled_dev_extensions = enabled_dev_extensions;
    vk->nb_enabled_dev_extensions = nb_enabled_dev_extensions;

    /* Queue family list (ffmpeg 7+ array API). */
#if WAYMUX_AVVK_HAVE_QF_ARRAY
    int n = 0;
    vk->qf[n].idx = (int)compute_qf;
    vk->qf[n].num = 1;
    vk->qf[n].flags = (VkQueueFlagBits)(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT);
    vk->qf[n].video_caps = (VkVideoCodecOperationFlagBitsKHR)0;
    n++;
    if (encode_qf != UINT32_MAX && encode_qf != compute_qf) {
        vk->qf[n].idx = (int)encode_qf;
        vk->qf[n].num = 1;
        vk->qf[n].flags = (VkQueueFlagBits)VK_QUEUE_VIDEO_ENCODE_BIT_KHR;
        vk->qf[n].video_caps = (VkVideoCodecOperationFlagBitsKHR)(
            VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR |
            VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR);
        n++;
    }
    vk->nb_qf = n;
#endif

    /*
     * Fixed queue_family_*_index fields. On ffmpeg 6 (libavutil < 59) these
     * are the primary, non-deprecated API and MUST be set. On ffmpeg 7/8 they
     * are deprecated-but-present under FF_API_VULKAN_FIXED_QUEUES, set for
     * backwards compatibility alongside the qf[] array above.
     */
#if !WAYMUX_AVVK_HAVE_QF_ARRAY || defined(FF_API_VULKAN_FIXED_QUEUES)
    vk->queue_family_index = -1;
    vk->nb_graphics_queues = 0;
    vk->queue_family_tx_index = (int)compute_qf;
    vk->nb_tx_queues = 1;
    vk->queue_family_comp_index = (int)compute_qf;
    vk->nb_comp_queues = 1;
    if (encode_qf != UINT32_MAX) {
        vk->queue_family_encode_index = (int)encode_qf;
        vk->nb_encode_queues = 1;
    } else {
        vk->queue_family_encode_index = -1;
        vk->nb_encode_queues = 0;
    }
    vk->queue_family_decode_index = -1;
    vk->nb_decode_queues = 0;
#endif

    /*
     * Deprecated FF_API_VULKAN_SYNC_QUEUES fields. Leave NULL — ffmpeg
     * fills in mutex-based implementations during ctx_init.
     */
#ifdef FF_API_VULKAN_SYNC_QUEUES
    vk->lock_queue = NULL;
    vk->unlock_queue = NULL;
#endif

    return 0;
}

/*
 * Configure an allocated-but-not-yet-init'd AVHWFramesContext (VULKAN).
 * Caller must have set format/sw_format/width/height already.
 *
 * tiling: 0 → VK_IMAGE_TILING_OPTIMAL (default).
 *         Use VK_IMAGE_TILING_LINEAR or DRM_FORMAT_MODIFIER as needed.
 * extra_usage: OR'd on top of the default usage flags ffmpeg applies.
 */
int waymux_avvk_set_frames(
    AVBufferRef *hw_frames_ref,
    uint32_t tiling,
    uint32_t extra_usage)
{
    if (!hw_frames_ref) return -1;
    AVHWFramesContext *fctx = (AVHWFramesContext *)hw_frames_ref->data;
    if (!fctx) return -2;
    AVVulkanFramesContext *vk = (AVVulkanFramesContext *)fctx->hwctx;
    if (!vk) return -3;

    vk->tiling = (VkImageTiling)tiling;
    vk->usage = (VkImageUsageFlagBits)(
        VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        extra_usage);
    vk->create_pnext = NULL;
    vk->flags = 0;
    vk->img_flags = 0;
    vk->nb_layers = 0;
    return 0;
}

/*
 * View an AVVkFrame (the pointer stored in AVFrame::data[0] when format
 * is AV_PIX_FMT_VULKAN).
 *
 * `plane` indexes into the AVVkFrame::img/mem/sem/access/layout/queue_family
 * arrays. For single-plane BGRA frames pass 0.
 */
typedef struct WaymuxVkFrameView {
    void     *img;
    void     *mem;
    void     *sem;
    uint64_t  sem_value;
    int32_t   layout;
    uint32_t  access;
    uint32_t  queue_family;
    uint32_t  flags;
    int32_t   tiling;
} WaymuxVkFrameView;

void waymux_avvk_frame_view(
    const void *avvkframe,
    WaymuxVkFrameView *out,
    int plane)
{
    if (!avvkframe || !out) return;
    const AVVkFrame *f = (const AVVkFrame *)avvkframe;
    out->img          = f->img[plane];
    out->mem          = f->mem[plane];
    out->sem          = f->sem[plane];
    out->sem_value    = f->sem_value[plane];
    out->layout       = (int32_t)f->layout[plane];
    out->access       = (uint32_t)f->access[plane];
    out->queue_family = f->queue_family[plane];
    out->flags        = (uint32_t)f->flags;
    out->tiling       = (int32_t)f->tiling;
}

/*
 * Update the post-submit state of an AVVkFrame plane. After waymux records
 * a vkCmdCopyImage into the frame's image, it must update the layout/access/
 * queue_family/sem_value here so the encoder's command buffer chains
 * onto the same timeline semaphore.
 */
void waymux_avvk_frame_update(
    void *avvkframe,
    int plane,
    uint64_t sem_value,
    int32_t  layout,
    uint32_t access,
    uint32_t queue_family)
{
    if (!avvkframe) return;
    AVVkFrame *f = (AVVkFrame *)avvkframe;
    f->sem_value[plane]    = sem_value;
    f->layout[plane]       = (VkImageLayout)layout;
    f->access[plane]       = (VkAccessFlagBits)access;
    f->queue_family[plane] = queue_family;
}
