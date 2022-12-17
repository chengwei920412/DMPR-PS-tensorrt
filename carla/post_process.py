import math, os, json
import cv2 as cv
import numpy as np
import torch

class PostProcess(object):
    def __init__(self, result_json):
        self.DIR_CLUSTER_TH = math.pi/20 #+-9 degree
        self.DIR_COVER_TH = math.pi/4
        self.QUADS = [deg*math.pi/180 for deg in [-360, -270, -180, -90, 0, 90, 180, 270, 360]]
        self.result_json = result_json
        self.mean_direction = 0.0

    def plot_points(self, image):
        """Plot marking points on the image."""
        if not self.result_json:
            return
        height = image.shape[0]
        width = image.shape[1]
        for mp in self.result_json["marking_points"]:
        # for confidence, marking_point in pred_points:
            p0_x = mp["p0x"]
            p0_y = mp["p0y"]
            cos_val = math.cos(mp["direction"])
            sin_val = math.sin(mp["direction"])
            dir_deg = mp["direction"] * 57.3
            p1_x = p0_x + 50*cos_val
            p1_y = p0_y + 50*sin_val
            p2_x = p0_x - 50*sin_val
            p2_y = p0_y + 50*cos_val
            p3_x = p0_x + 50*sin_val
            p3_y = p0_y - 50*cos_val
            p0_x = int(round(p0_x))
            p0_y = int(round(p0_y))
            p1_x = int(round(p1_x))
            p1_y = int(round(p1_y))
            p2_x = int(round(p2_x))
            p2_y = int(round(p2_y))
            cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
            mp_str = "{:.2f}, {:.2f}".format(mp["conf"], dir_deg)
            cv.putText(image, mp_str, (p0_x, p0_y),
                    cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255))
            # L-shape
            if mp["type"] > 0.5:
                cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
            # T-shape
            else:
                p3_x = int(round(p3_x))
                p3_y = int(round(p3_y))
                cv.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)

    def calc_mean_direction(self):
        mps = sorted(self.result_json["marking_points"], key=lambda x:x["direction"])
        # print(mps)
        cluster_list = []
        for mp0 in mps:
            # print("direction: {}".format(mp0["direction"]))
            dir0 = mp0["direction"]
            for mp in mps:
                if mp["idx"] == mp0["idx"]:
                    continue
                for quad in self.QUADS:
                    dir_diff = abs(mp["direction"] + quad - dir0)
                    if dir_diff < self.DIR_CLUSTER_TH:
                        mp["dir_quad"] = mp["direction"] + quad
                        cluster_list.append(mp)
                        break
            print("cluster on mp[{}]: {}".format(mp0["idx"], len(cluster_list)))
            if len(cluster_list) < len(mps)/2:
                print("mp[{}] is outlier, continue".format(mp0["idx"]))
                cluster_list.clear()
                continue
            else:
                print("mp[{}] cluster done".format(mp0["idx"]))
                break

        if len(cluster_list) > 0:
            for mp in cluster_list:
                self.mean_direction += mp["dir_quad"]
            self.mean_direction /= len(cluster_list)
            return True
        else:
            return False
        
    def fix_direction(self):
        for mp in self.result_json["marking_points"]:
            dir_diff_min = math.pi
            quad = 0.0
            for qd in self.QUADS:
                dir = self.mean_direction + qd
                # if dir > math.pi or dir < -math.pi:
                #     pass
                dir_diff = abs(dir - mp["direction"])
                if dir_diff < dir_diff_min:
                    dir_diff_min = dir_diff
                    quad = qd 

            print("dir_diff_min: {:.2f}, quad: {:.2f}".format(dir_diff_min, quad))
            if dir_diff_min < self.DIR_COVER_TH:
                dir = self.mean_direction + quad
                if dir > math.pi:
                    dir -= 2*math.pi
                elif dir < -math.pi:
                    dir += 2*math.pi
                print("set mp[{}] direction from {:.2f} to {:.2f}".format(mp["idx"], mp["direction"], dir))
                mp["direction"] = dir
