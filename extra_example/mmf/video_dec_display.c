#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/param.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <inttypes.h>
#include <errno.h>

#include "cvi_sys.h"
#include <linux/cvi_type.h>
#include "cvi_buffer.h"
#include "cvi_ae_comm.h"
#include "cvi_awb_comm.h"
#include "cvi_comm_isp.h"

#include "sample_comm.h"

#define VDEC_WIDTH 1080
#define VDEC_HEIGHT 1920
#define SAMPLE_STREAM_PATH "./"
#define VDEC_BIND_VPSS_VENC_DELAY 60000
#define DEFAULT_MAX_FRM_BUF 4

static bool is_running = true;
static pthread_t send_vo_thread;

static void initVdecThreadParam(
		vdecChnCtx *pvdchnCtx,
		VDEC_THREAD_PARAM_S *pvtp,
		char *path,
		CVI_S32 s32MilliSec)
{
	SAMPLE_VDEC_ATTR *psvdattr = &pvdchnCtx->stSampleVdecAttr;

	snprintf(pvtp->cFileName,
			sizeof(pvtp->cFileName), path);
	snprintf(pvtp->cFilePath,
			sizeof(pvtp->cFilePath), "%s",
			SAMPLE_STREAM_PATH);

	pvtp->enType = psvdattr->enType;
	pvtp->s32StreamMode = psvdattr->enMode;
	pvtp->s32ChnId = pvdchnCtx->VdecChn;
	pvtp->s32IntervalTime = 10000;
	pvtp->u64PtsInit = 0;
	pvtp->u64PtsIncrease = 1;
	pvtp->eThreadCtrl = THREAD_CTRL_START;
	pvtp->bCircleSend = CVI_FALSE;
	// pvtp->bCircleSend = CVI_TRUE;
	pvtp->s32MilliSec = s32MilliSec;
	pvtp->s32MinBufSize = (psvdattr->u32Width * psvdattr->u32Height * 3) >> 1;
	pvtp->bFileEnd = CVI_FALSE;
}

int alarm_sockets[2];
CVI_VOID *thread_send_vo(CVI_VOID *arg)
{
	CVI_S32 s32Ret = CVI_SUCCESS;
	CVI_BOOL bFirstFrame = CVI_FALSE;
	VDEC_THREAD_PARAM_S *pstVdecThreadParam = (VDEC_THREAD_PARAM_S *)arg;
	VIDEO_FRAME_INFO_S stVdecFrame = {0};
	VIDEO_FRAME_INFO_S stOverlayFrame = {0};
	VPSS_GRP VpssGrp0 = 0;
	VPSS_CHN VpssChn = 0;
	VO_LAYER VoLayer = 0;
	VO_CHN VoChn = 0;
	int retry = 0;
	int frm_cnt = 0;
#ifdef SHOW_STATISTICS_2
	struct timeval tv1, tv2;
#endif
	fd_set readfds;
	struct timeval tv;

	prctl(PR_SET_NAME, "thread_send_vo");
#ifdef SHOW_STATISTICS_2
	tv1.tv_usec = tv1.tv_sec = 0;
#endif
	while (pstVdecThreadParam->eThreadCtrl == THREAD_CTRL_START) {
		int fail = 0;
		int r;

		FD_ZERO(&readfds);
		FD_SET(alarm_sockets[1], &readfds);
		tv.tv_sec = 1;
		tv.tv_usec = 0;
		r = select(alarm_sockets[1] + 1, &readfds, NULL, NULL, &tv);
		if (r == -1) {
			if (errno == EINTR) {
				fprintf(stderr, "%s: select interrupt\n", __func__);
				continue;
			}
			continue;
		} else if (r == 0) {
			struct itimerval cur;

			getitimer(ITIMER_REAL, &cur);
			fprintf(stderr, "%s: select timeout. %ld %d\n", __func__
				, cur.it_value.tv_usec, frm_cnt);
			// continue;
		} else {
			char buf[240];
			int ret = read(alarm_sockets[1], buf, sizeof(buf));

			if (ret != 4)
				fprintf(stderr, "select read %d\n", ret);
		}

		bFirstFrame = CVI_TRUE;
		if (!bFirstFrame) {
			s32Ret = CVI_VPSS_SendChnFrame(VpssGrp0, VpssChn, &stOverlayFrame, 1000);
			if (s32Ret != CVI_SUCCESS) {
				CVI_VPSS_ReleaseChnFrame(VpssGrp0, VpssChn, &stOverlayFrame);
				continue;
			}
			CVI_VPSS_ReleaseChnFrame(VpssGrp0, VpssChn, &stOverlayFrame);
		}
RETRY_GET_FRAME:
		s32Ret = CVI_VDEC_GetFrame(pstVdecThreadParam->s32ChnId, &stVdecFrame, 1000);
		if (s32Ret != CVI_SUCCESS) {
			// printf("[%d] CVI_VDEC_GetFrame fail\n", pstVdecThreadParam->s32ChnId);
			fail++;
			retry++;
			if (s32Ret == CVI_ERR_VDEC_BUSY) {
				printf("get frame timeout ..in overlay ..retry\n");
				goto RETRY_GET_FRAME;
			}
			// break;
			continue;
		}

		s32Ret = CVI_VPSS_SendFrame(VpssGrp0, &stVdecFrame, 1000);

		if (s32Ret != CVI_SUCCESS) {
			CVI_VDEC_ReleaseFrame(pstVdecThreadParam->s32ChnId, &stVdecFrame);
			continue;
		}

		s32Ret = CVI_VPSS_GetChnFrame(VpssGrp0, VpssChn, &stOverlayFrame, 1000);
		CVI_VDEC_ReleaseFrame(pstVdecThreadParam->s32ChnId, &stVdecFrame);
		if (s32Ret != CVI_SUCCESS) {
			printf("CVI_VPSS_GetChnFrame faile, grp:%d\n", VpssGrp0);
			continue;
		}
		bFirstFrame = CVI_FALSE;
		// }
		if (fail > 0) {
			// usleep(10000);
			continue;
		}

		// if (is_using_vo) {
		s32Ret = CVI_VO_SendFrame(VoLayer, VoChn, &stOverlayFrame, 1000);
		if (s32Ret != CVI_SUCCESS) {
			printf("CVI_VO_SendFrame faile\n");
		}
		// }
		frm_cnt++;
		CVI_VPSS_ReleaseChnFrame(VpssGrp0, VpssChn, &stOverlayFrame);
#ifdef SHOW_STATISTICS_2
		gettimeofday(&tv2, NULL);
		int curr_ms =
			(tv2.tv_sec - tv1.tv_sec) * 1000 + (tv2.tv_usec/1000) - (tv1.tv_usec/1000);
		printf("CVI_VO_SendFrame delta %d ms try %d times\n", curr_ms, retry);
		tv1 = tv2;
#endif
		retry = 0;
	}
	printf("thread_send_vo exit\n");
	return NULL;
}

CVI_S32 start_send_vo_thread(VDEC_THREAD_PARAM_S *pstVdecThreadParam)
{
	CVI_S32 s32Ret = CVI_SUCCESS;
	struct sched_param param;
	pthread_attr_t attr;

	param.sched_priority = 80;
	pthread_attr_init(&attr);
	pthread_attr_setschedpolicy(&attr, SCHED_RR);
	pthread_attr_setschedparam(&attr, &param);
	pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);

	s32Ret = pthread_create(&send_vo_thread, &attr, thread_send_vo, (void *)pstVdecThreadParam);
	if (s32Ret != 0) {
		return CVI_FAILURE;
	}

	return CVI_SUCCESS;
}

CVI_VOID stop_send_vo_thread(VDEC_THREAD_PARAM_S *pstVdecThreadParam)
{
	pstVdecThreadParam->eThreadCtrl = THREAD_CTRL_STOP;
	if (send_vo_thread != 0)
		pthread_join(send_vo_thread, NULL);
}

CVI_S32 VDEC_DISPLAY(char *video_path)
{
	CVI_S32 s32Ret = CVI_SUCCESS;

	COMPRESS_MODE_E    enCompressMode   = COMPRESS_MODE_NONE;
	VB_CONFIG_S        stVbConf;
	CVI_U32	       u32BlkSize;
	SIZE_S stSize;
	stSize.u32Width = VDEC_WIDTH;
	stSize.u32Height = VDEC_HEIGHT;

	/************************************************
	 * Init SYS and common VB
	 ************************************************/
	memset(&stVbConf, 0, sizeof(VB_CONFIG_S));
	stVbConf.u32MaxPoolCnt		= 1;

	u32BlkSize = COMMON_GetPicBufferSize(stSize.u32Width, stSize.u32Height, PIXEL_FORMAT_RGB_888, DATA_BITWIDTH_8
					    , enCompressMode, DEFAULT_ALIGN);
	stVbConf.astCommPool[0].u32BlkSize	= u32BlkSize;
	stVbConf.astCommPool[0].u32BlkCnt	= 3;
	SAMPLE_PRT("common pool[0] BlkSize %d\n", u32BlkSize);

	s32Ret = SAMPLE_COMM_SYS_Init(&stVbConf);
	if (s32Ret != CVI_SUCCESS) {
		SAMPLE_PRT("system init failed with %#x\n", s32Ret);
		return -1;
	}

	/************************************************
	 * Init VPSS
	 ************************************************/
	VPSS_GRP	   VpssGrp0	  = 0;
	VPSS_GRP_ATTR_S    stVpssGrpAttr;
	VPSS_CHN           VpssChn        = VPSS_CHN0;
	CVI_BOOL           abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
	VPSS_CHN_ATTR_S    astVpssChnAttr[VPSS_MAX_PHY_CHN_NUM] = {0};

	stVpssGrpAttr.stFrameRate.s32SrcFrameRate    = -1;
	stVpssGrpAttr.stFrameRate.s32DstFrameRate    = -1;
	stVpssGrpAttr.enPixelFormat                  = PIXEL_FORMAT_YUV_PLANAR_420;
	stVpssGrpAttr.u32MaxW                        = VDEC_WIDTH;
	stVpssGrpAttr.u32MaxH                        = VDEC_HEIGHT;
	stVpssGrpAttr.u8VpssDev                      = 0;

	astVpssChnAttr[VpssChn].u32Width                    = VDEC_WIDTH;
	astVpssChnAttr[VpssChn].u32Height                   = VDEC_HEIGHT;
	astVpssChnAttr[VpssChn].enVideoFormat               = VIDEO_FORMAT_LINEAR;
	astVpssChnAttr[VpssChn].enPixelFormat               = PIXEL_FORMAT_RGB_888;
	astVpssChnAttr[VpssChn].stFrameRate.s32SrcFrameRate = -1;
	astVpssChnAttr[VpssChn].stFrameRate.s32DstFrameRate = -1;
	astVpssChnAttr[VpssChn].u32Depth                    = 0;
	astVpssChnAttr[VpssChn].bMirror                     = CVI_FALSE;
	astVpssChnAttr[VpssChn].bFlip                       = CVI_FALSE;
	astVpssChnAttr[VpssChn].stAspectRatio.enMode        = ASPECT_RATIO_NONE;
	astVpssChnAttr[VpssChn].stNormalize.bEnable         = CVI_FALSE;

	/*start vpss*/
	abChnEnable[0] = CVI_TRUE;
	s32Ret = SAMPLE_COMM_VPSS_Init(VpssGrp0, abChnEnable, &stVpssGrpAttr, astVpssChnAttr);
	if (s32Ret != CVI_SUCCESS) {
		printf("init vpss group failed. s32Ret: 0x%x !\n", s32Ret);
		return s32Ret;
	}

	s32Ret = SAMPLE_COMM_VPSS_Start(VpssGrp0, abChnEnable, &stVpssGrpAttr, astVpssChnAttr);
	if (s32Ret != CVI_SUCCESS) {
		printf("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
		return s32Ret;
	}

	/************************************************
	 * Init VO
	 ************************************************/
	SAMPLE_VO_CONFIG_S stVoConfig;
	RECT_S stDefDispRect  = {0, 0, 1080, 1920};
	SIZE_S stDefImageSize = {1080, 1920};
	VO_CHN VoChn = 0;

	s32Ret = SAMPLE_COMM_VO_GetDefConfig(&stVoConfig);
	if (s32Ret != CVI_SUCCESS) {
		printf("SAMPLE_COMM_VO_GetDefConfig failed with %#x\n", s32Ret);
		return s32Ret;
	}

	stVoConfig.VoDev	 = 0;
	stVoConfig.stVoPubAttr.enIntfType  = VO_INTF_MIPI;
	stVoConfig.stVoPubAttr.enIntfSync  = VO_OUTPUT_1080P30;
	stVoConfig.stDispRect	 = stDefDispRect;
	stVoConfig.stImageSize	 = stDefImageSize;
	stVoConfig.enPixFormat	 = PIXEL_FORMAT_RGB_888;
	stVoConfig.enVoMode	 = VO_MODE_1MUX;

	s32Ret = SAMPLE_COMM_VO_StartVO(&stVoConfig);
	if (s32Ret != CVI_SUCCESS) {
		printf("SAMPLE_COMM_VO_StartVO failed with %#x\n", s32Ret);
		return s32Ret;
	}

	/************************************************
	 *  Init VDEC VB  and  start VDEC
	 ************************************************/
	vdecChnCtx chnCtx0 = {0};

	chnCtx0.VdecChn = 0;
	chnCtx0.stSampleVdecAttr.enType = PT_H264;
	chnCtx0.stSampleVdecAttr.u32Width = VDEC_WIDTH;
	chnCtx0.stSampleVdecAttr.u32Height = VDEC_HEIGHT;
	chnCtx0.stSampleVdecAttr.enMode = VIDEO_MODE_FRAME;
	chnCtx0.stSampleVdecAttr.enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
	chnCtx0.stSampleVdecAttr.u32DisplayFrameNum = 3;
	chnCtx0.stSampleVdecAttr.u32FrameBufCnt = DEFAULT_MAX_FRM_BUF << 1;
	chnCtx0.bCreateChn = CVI_FALSE;

	SAMPLE_COMM_VDEC_SetVbMode(3);
	SAMPLE_VDEC_ATTR astSampleVdec[VDEC_MAX_CHN_NUM];

	astSampleVdec[0].enType = PT_H264;
	astSampleVdec[0].u32Width = VDEC_WIDTH;
	astSampleVdec[0].u32Height = VDEC_HEIGHT;
	astSampleVdec[0].enMode = VIDEO_MODE_FRAME;
	astSampleVdec[0].stSampleVdecVideo.enDecMode = VIDEO_DEC_MODE_IP;
	astSampleVdec[0].stSampleVdecVideo.enBitWidth = DATA_BITWIDTH_8;
	astSampleVdec[0].stSampleVdecVideo.u32RefFrameNum = 2;
	astSampleVdec[0].u32DisplayFrameNum = 3;
	astSampleVdec[0].u32FrameBufCnt = DEFAULT_MAX_FRM_BUF << 1;
	astSampleVdec[0].enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;

	s32Ret = SAMPLE_COMM_VDEC_InitVBPool(1, &astSampleVdec[0]);
	if (s32Ret != CVI_SUCCESS) {
		CVI_VDEC_ERR("SAMPLE_COMM_VDEC_InitVBPool fail\n");
	}

	s32Ret = SAMPLE_COMM_VDEC_Start(&chnCtx0);
	if (s32Ret != CVI_SUCCESS) {
		printf("start VDEC fail for %#x!\n", s32Ret);
		goto out;
	}

    /************************************************
    *  send stream to VDEC
    *************************************************/
	initVdecThreadParam(&chnCtx0, &chnCtx0.stVdecThreadParamSend, video_path, -1);
	SAMPLE_COMM_VDEC_StartSendStream(&chnCtx0.stVdecThreadParamSend, &chnCtx0.vdecThreadSend);

	usleep(100000);
	start_send_vo_thread(&chnCtx0.stVdecThreadParamSend);

	do {
		printf("---------------press Ctrl+C to exit!---------------\n");
		getchar();
	}while(is_running);

	stop_send_vo_thread(&chnCtx0.stVdecThreadParamSend);
	SAMPLE_COMM_VDEC_CmdCtrl(&chnCtx0.stVdecThreadParamSend, &chnCtx0.vdecThreadSend);
	SAMPLE_COMM_VDEC_StopSendStream(&chnCtx0.stVdecThreadParamSend, &chnCtx0.vdecThreadSend);

out:
	SAMPLE_COMM_VO_StopVO(&stVoConfig);
	SAMPLE_COMM_VPSS_Stop(VpssGrp0, abChnEnable);
	SAMPLE_COMM_VDEC_Stop(1);
	SAMPLE_COMM_VDEC_ExitVBPool();
	SAMPLE_COMM_SYS_Exit();
	return s32Ret;
}

unsigned int setalarm(unsigned int msec)
{
	struct itimerval old, new;

	new.it_interval.tv_usec = msec*1000;
	new.it_interval.tv_sec = 0;
	new.it_value.tv_usec = 0;
	new.it_value.tv_sec = 1;
	if (setitimer(ITIMER_REAL, &new, &old) < 0)
		return 0;
	else
		return old.it_value.tv_sec;
}

CVI_VOID EXAMPLE_VDEC_HandleSig(CVI_S32 signo)
{
	if (signo == SIGALRM) {
		char buf[4] = "qoo\n";

		write(alarm_sockets[0], buf, 4);
	} else if (SIGINT == signo || SIGTSTP == signo || SIGTERM == signo) {
		//release
		is_running = 0;
		printf("program exit normally!\n");
	}
}

int main(int argc, char *argv[])
{
	CVI_S32 s32Ret = CVI_FAILURE;
	CVI_S32 s32Index;

    if (argc != 2)
    {
        printf("%s <1080P h264_path> \n", argv[0]);
        printf("Usage: %s  test.h264 \n", argv[0]);
        return -1;
    }

	signal(SIGINT, EXAMPLE_VDEC_HandleSig);
	signal(SIGALRM, EXAMPLE_VDEC_HandleSig);
	signal(SIGTERM, EXAMPLE_VDEC_HandleSig);

	socketpair(AF_UNIX, SOCK_STREAM, 0, alarm_sockets);
	setalarm(40);

	s32Ret = VDEC_DISPLAY(argv[1]);
	if (s32Ret == CVI_SUCCESS) {
		printf("program exit normally!\n");
	} else {
		printf("program exit abnormally!\n");
	}

	close(alarm_sockets[0]);
	close(alarm_sockets[1]);
	return s32Ret;
}

