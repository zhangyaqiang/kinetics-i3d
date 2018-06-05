import cv2
import os, glob
from multiprocessing import Pool, current_process

import argparse
out_path = ''


def dump_frames(vid_path):
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in range(fcount):
        ret, frame = video.read()
        assert ret
        cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)
        access_path = '{}/{:06d}.jpg'.format(vid_name, i)
        file_list.append(access_path)
    print('{} done'.format(vid_name))
    return file_list


def run_optical_flow(vid_item, dev_id=0):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = vid_item[0][:vid_item[0].rfind('.')]
    out_full_path = out_full_path.replace('Moments_in_Time_Mini_25fps', 'Moments_in_Time_Mini_flow')

    # if os.path.exists(out_full_path):
    #     return True

    try:
        os.makedirs(out_full_path)
    except OSError:
        pass

    current = current_process()
    # dev_id = int(current._identity[0]) - 1
    image_path = '{}/img'.format(out_full_path)
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = '/home/zyq/moment_in_time/dense_flow/build/extract_gpu -f={} -x={} -y={} -i={} -b=20 -t=1 -d={} -s=1 -o=dir'.format(vid_path, flow_x_path, flow_y_path, image_path, dev_id)

    os.system(cmd)
    print ('{} {} done'.format(vid_id, vid_name))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    # parser.add_argument("src_dir")
    # parser.add_argument("out_dir")
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--flow_type", type=str, default='tvl1', choices=['tvl1', 'warp_tvl1'])

    args = parser.parse_args()

    # out_path = args.out_dir
    # src_path = args.src_dir
    num_worker = args.num_worker
    flow_type = args.flow_type

    dataset_list = ['/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/validation',
                     '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training']

    # vid_list = []
    # for dataset_path in dataset_list:
    #     class_list = glob.glob(os.path.join(dataset_path, '*'))
    #     for class_path in class_list:
    #         video_list = glob.glob(os.path.join(class_path, '*mp4'))
    #         vid_list.extend(video_list)

    vid_list = ['/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/cooking/flickr-2-0-5-7-0-2-3-3-4720570233_26.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/sliding/getty-stop-motion-animated-dog-sledding-down-a-snowy-hill-to-crash-into-the-video-id129172768_2.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/dining/getty-college-student-friends-learning-and-teaching-together-in-a-video-id628631104_15.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/dining/vine-Vine-by-AP-Dining-OKUMtlvwADi_3.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/knitting/yt-V803qMcMJo8_622.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/knitting/yt-Pslwa3O2ZF4_817.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/destroying/yt-2raKGCdV7YA_622.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/destroying/yt-CVnKkYPDaPg_6.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/brushing/getty-handmade-thai-style-umbrella-painting-video-id451652677_2.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/brushing/yt-3RX-mER087Q_14.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/protesting/getty-aerial-shot-police-surrounded-by-students-students-attacking-police-video-id178696378_23.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/protesting/getty-syrian-security-forces-have-fired-on-a-mass-protest-of-thousands-in-video-id112715346_33.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/playing+music/yt-2ZSyrgxfPnU_35.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/playing+music/yt-2_I3Qp5xmps_145.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/dancing/yt-dTXvFkAr73k_65.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/dancing/yt-C6xt8OvGoTM_328.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/bouncing/getty-slow-motion-male-soccer-player-kicking-bouncing-ball-england-video-id442-22_1.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/bouncing/yt-cdDyIxjBQIc_40.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/playing+sports/getty-friends-have-fun-at-go-cart-video-id607727462_1.mp4',
                '/home/zyq/moment_in_time/Moments_in_Time_Mini_25fps/training/playing+sports/flickr-0-5-4-6-9-2-9-0-4705469290_1.mp4']

    pool = Pool(num_worker)
    # if flow_type == 'tvl1':
    pool.map(run_optical_flow, zip(vid_list, range(len(vid_list))))
    # elif flow_type == 'warp_tvl1':
    #     pool.map(run_warp_optical_flow, zip(vid_list, range(len(vid_list))))