"""Inference demo of directional point detector."""
import math, os, json
import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import ToTensor
import config
from data import get_predicted_points, pair_marking_points, calc_point_squre_dist, pass_through_third_point
from model import DirectionalPointDetector
from util import Timer
from carla.post_process import PostProcess

def preprocess_image(image):
    """Preprocess numpy image to torch tensor."""
    if image.shape[0] != 512 or image.shape[1] != 512:
        image = cv.resize(image, (512, 512))
    return torch.unsqueeze(ToTensor()(image), 0)

def detect_marking_points(detector, image, thresh, device):
    """Given image read from opencv, return detected marking points."""
    prediction = detector(preprocess_image(image).to(device))
    return get_predicted_points(prediction[0], thresh)

def detect_carla_image(detector, device, args, carla_ipm_dir):
    output_dir = "./output/"
    files = os.listdir(carla_ipm_dir)
    counter = 0
    results_list = []
    for fname in files:
        fullpath = carla_ipm_dir + "/" + fname
        print("evaluate [{}] {}".format(counter, fullpath))
        output_fullpath = output_dir + "/" + fname

        image = cv.imread(fullpath)
        pred_points = detect_marking_points(detector, image, args.thresh, device)
        # slots = None
        if pred_points and args.inference_slot:
            pass
            # marking_points = list(list(zip(*pred_points))[1])
            # slots = inference_slots(marking_points)

        # format json
        result = {}
        result["counter"] = counter
        result["img_path"] = fullpath
        result["height"] = int(image.shape[0])
        result["width"] = int(image.shape[1])
        marking_points = []
        idx = 0
        for conf, marking_point in pred_points:
            mp = {}
            mp["idx"] = idx
            idx += 1
            mp["conf"] = float(conf)
            mp["shape"] = float(marking_point.shape)
            if mp["shape"] > 0.5:
                mp["type"] = "L"
            else:
                mp["type"] = "T"
            mp["x"] = float(marking_point.x)
            mp["y"] = float(marking_point.y)
            mp["direction"] = float(marking_point.direction)
            mp["p0x"] = int(result["width"] * mp["x"] - 0.5)
            mp["p0y"] = int(result["height"] * mp["y"] - 0.5)
            marking_points.append(mp)
        result["marking_points"] = marking_points
        results_list.append(result)
        post_processor = PostProcess(result)
        res = post_processor.calc_mean_direction()
        if res:
            post_processor.fix_direction()
        post_processor.add_branches()
        post_processor.infer_slots()
        post_processor.plot_points(image)
        post_processor.plot_slots(image)

        counter += 1
        # plot_points(image, result)
        # plot_slots(image, pred_points, slots)
        # cv.imshow('demo', image)
        # cv.waitKey(1)
        # if args.save:
        cv.imwrite(output_fullpath, image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    # print(results_list)

    # save json
    json_file_path = output_dir + "/results.json"
    with open(json_file_path, 'wt') as fd:
        for dict in results_list:
            json.dump(dict, fd, ensure_ascii=False)
            fd.write("\n")

def main(args):
    """Inference demo of directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    torch.set_grad_enabled(False)
    dp_detector = DirectionalPointDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    dp_detector.load_state_dict(torch.load(args.detector_weights, map_location=torch.device('cpu')))
    dp_detector.eval()
    detect_carla_image(dp_detector, device, args, carla_ipm_dir="./dataset/carla")

if __name__ == '__main__':
    args = config.get_parser_for_inference().parse_args()
    args.detector_weights = "weights/dmpr_pretrained_weights.pth"
    args.inference_slot = True
    main(args)
   
