import math, os, json
import cv2
import numpy as np
import torch

class PostProcess(object):
    def __init__(self, result_json):
        self.SLOT_LENGTH_M = 5.9
        self.SLOT_WIDTH_M = 3.75
        self.PPM = 640/18.0
        self.SLOT_LENGTH = self.SLOT_LENGTH_M * self.PPM
        self.SLOT_WIDTH = self.SLOT_WIDTH_M * self.PPM
        self.VERTEX_TH = self.SLOT_WIDTH / 2.0
        self.DIR_CLUSTER_TH = math.pi/20 #+-9 degree
        self.BRANCH_PAIR_TH = math.pi/8
        self.DIR_COVER_TH = math.pi/4
        self.QUADS = [deg*math.pi/180 for deg in [-360, -270, -180, -90, 0, 90, 180, 270, 360]]
        self.result_json = result_json
        self.mean_direction = 0.0
        self.virtual_vertex = []

    def plot_slots(self, image):
        print("total slots: {}".format(len(self.result_json["slots"])))
        for slot in self.result_json["slots"]:
            vertices = []
            pt0 = (slot["p0x"], slot["p0y"])
            pt1 = (slot["p1x"], slot["p1y"])
            pt2 = (slot["p2x"], slot["p2y"])
            pt3 = (slot["p3x"], slot["p3y"])
            vertices.append(pt0)
            vertices.append(pt1)
            vertices.append(pt2)
            vertices.append(pt3)
            cx = int((pt0[0]+pt1[0]+pt2[0]+pt3[0])/4.0)
            cy = int((pt0[1]+pt1[1]+pt2[1]+pt3[1])/4.0)
            if cx < 20:
                cx = 20
            if cy < 20:
                cy = 20
            
            cv2.putText(
                img = image,
                text = "{:d}".format(slot["idx"]),
                org = (cx, cy),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 1.0,
                color = (0, 255, 255),
                thickness = 2
            )
            for i in range(4):
                cv2.line(
                    img = image, 
                    pt1 = vertices[i], 
                    pt2 = vertices[(i+1)%4], 
                    color = (0, 255, 0),
                    thickness = 3,
                    lineType = cv2.LINE_8
                    )

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
            # cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
            mp_str = "{:.2f}".format(mp["conf"])
            cv2.putText(image, mp_str, (p0_x, p0_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 255))
            
            # L-shape
            if mp["shape"] > 0.5:
                # cv2.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
                cv2.circle(
                    img=image, 
                    center=(p0_x, p0_y),
                    radius=12, 
                    color=(0,255,255), 
                    thickness=6
                )
            # T-shape
            else:
                cv2.circle(
                    img=image, 
                    center=(p0_x, p0_y),
                    radius=12, 
                    color=(0,0,255), 
                    thickness=6
                )
                # p3_x = int(round(p3_x))
                # p3_y = int(round(p3_y))
                # cv2.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)

    def calc_mean_direction(self):
        mps = sorted(self.result_json["marking_points"], key=lambda x:x["direction"])
        # print(mps)
        cluster_list = []
        for mp0 in mps:
            cluster_list.clear()
            # print("direction: {}".format(mp0["direction"]))
            dir0 = mp0["direction"]
            mp0["dir_quad"] = dir0
            cluster_list.append(mp0)
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
            if len(cluster_list) < len(mps)/2+1:
                print("mp[{}] is outlier, continue".format(mp0["idx"]))
                continue
            else:
                print("mp[{}] cluster done".format(mp0["idx"]))
                for mp in cluster_list:
                    self.mean_direction += mp["dir_quad"]
                self.mean_direction /= len(cluster_list)
                return True

        return False

    def set_direction(self, dir):
        if dir > math.pi:
            dir -= 2*math.pi
        elif dir < -math.pi:
            dir += 2*math.pi
        else:
            pass
        return dir

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
                dir = self.set_direction(self.mean_direction + quad)
                print("set mp[{}] direction from {:.2f} to {:.2f}".format(mp["idx"], mp["direction"], dir))
                mp["direction"] = dir

    def add_branches(self):
        self.virtual_vertex = []
        # add branches to each marking-point
        for mp in self.result_json["marking_points"]:
            bf = {}
            bf["name"] = "forward"
            bf["dir"] = mp["direction"]
            bf["cos"] = math.cos(bf["dir"])
            bf["sin"] = math.sin(bf["dir"])
            bf["p1x"] = mp["p0x"] + bf["cos"]*self.SLOT_LENGTH
            bf["p1y"] = mp["p0y"] + bf["sin"]*self.SLOT_LENGTH
            bf["paired"] = False
            mp["branch_forward"] = bf
            self.virtual_vertex.append([bf["p1x"], bf["p1y"]])

            br = {}
            br["name"] = "right"
            br["dir"] = self.set_direction(mp["direction"] + math.pi/2)
            br["cos"] = math.cos(br["dir"])
            br["sin"] = math.sin(br["dir"])
            br["p2x"] = bf["p1x"] + br["cos"]*self.SLOT_WIDTH
            br["p2y"] = bf["p1y"] + br["sin"]*self.SLOT_WIDTH
            br["p3x"] = mp["p0x"] + br["cos"]*self.SLOT_WIDTH
            br["p3y"] = mp["p0y"] + br["sin"]*self.SLOT_WIDTH
            br["paired"] = False
            mp["branch_right"] = br

            if mp["type"] == "T":
                bl = {}
                bl["name"] = "left"
                bl["dir"] = self.set_direction(mp["direction"] - math.pi/2)
                bl["cos"] = math.cos(bl["dir"])
                bl["sin"] = math.sin(bl["dir"])
                bl["p2x"] = bf["p1x"] + bl["cos"]*self.SLOT_WIDTH
                bl["p2y"] = bf["p1y"] + bl["sin"]*self.SLOT_WIDTH
                bl["p3x"] = mp["p0x"] + bl["cos"]*self.SLOT_WIDTH
                bl["p3y"] = mp["p0y"] + bl["sin"]*self.SLOT_WIDTH
                bl["paired"] = False
                mp["branch_left"] = bl

    # search marking-point in radius: self.VERTEX_TH
    def find_slot_vertex(self, px, py):
        dist_min = self.result_json["width"]
        mp_found = None
        for mp in self.result_json["marking_points"]:
            p0x = mp["p0x"]
            p0y = mp["p0y"]
            dist = math.sqrt((px-p0x)**2 + (py-p0y)**2)
            if dist<self.VERTEX_TH and dist<dist_min:
                dist_min = dist
                mp_found = mp
        return mp_found

    # virtual vertex added by branches
    def find_virtual_vertex(self, px, py):
        dist_min = self.result_json["width"]
        vv_found = None
        for vv in self.virtual_vertex:
            p0x = vv[0]
            p0y = vv[1]
            dist = math.sqrt((px-p0x)**2 + (py-p0y)**2)
            if dist<self.VERTEX_TH and dist<dist_min:
                dist_min = dist
                vv_found = vv
        return vv_found        

    def set_paired_branch(self, branch, mp):
        bf = mp["branch_forward"]
        br = mp["branch_right"]
        if mp.__contains__("branch_left"):
            bl = mp["branch_left"]

        rad_diff = abs(branch["dir"] - bf["dir"])
        if abs(rad_diff - math.pi) < self.BRANCH_PAIR_TH:
            if bf["paired"]:
                print("WTF! branch forward already paired!")
                return False
            bf["paired"] = True
            br["paired"] = True
            if mp.__contains__("branch_left"):

                bl["paired"] = True
            return True
        
        rad_diff = abs(branch["dir"] - br["dir"])
        if abs(rad_diff - math.pi) < self.BRANCH_PAIR_TH:
            if br["paired"]:
                print("WTF! branch right already paired!")
                return False
            br["paired"] = True
            return True

        if mp.__contains__("branch_left"):
            bl = mp["branch_left"]
            rad_diff = abs(branch["dir"] - bl["dir"])
            if abs(rad_diff - math.pi) < self.BRANCH_PAIR_TH:
                if bl["paired"]:
                    print("WTF! branch left already paired!")
                    return False
                bl["paired"] = True
                return True
        return False

    # search p1
    def infer_branch_forward(self, bf):
        p1x = bf["p1x"]
        p1y = bf["p1y"]
        mp = self.find_slot_vertex(p1x, p1y)
        if mp:
            bf["p1x"] = mp["p0x"]
            bf["p1y"] = mp["p0y"]
            # assert self.set_paired_branch(bf, mp), "pair branch failed!"
            if not self.set_paired_branch(bf, mp):
                print("WTF! pair branch forward failed")
                return False
        return True

    # search p2 && p3
    def infer_branch_right_left(self, brl):
        p2x = brl["p2x"]
        p2y = brl["p2y"]
        mp = self.find_slot_vertex(p2x, p2y)
        if mp:
            brl["p2x"] = mp["p0x"]
            brl["p2y"] = mp["p0y"]
            # assert self.set_paired_branch(brl, mp), "pair branch failed!"
            if not self.set_paired_branch(brl, mp):
                print("WTF! pair branch right or left failed")
                return False
        else:
            vv = self.find_virtual_vertex(p2x, p2y)
            if vv:
                print("find virtual vertex: [{:.1f}, {:.1f}]".format(vv[0], vv[1]))
                brl["p2x"] = vv[0]
                brl["p2y"] = vv[1]

        p3x = brl["p3x"]
        p3y = brl["p3y"]
        mp = self.find_slot_vertex(p3x, p3y)
        if mp:
            brl["p3x"] = mp["p0x"]
            brl["p3y"] = mp["p0y"]
            # assert self.set_paired_branch(brl, mp), "pair branch failed!"
            if not self.set_paired_branch(brl, mp):
                print("WTF! pair branch right or left failed")
                return False
        return True

    def infer_slots(self):
        self.result_json["slots"] = []
        counter = 0
        for mp in self.result_json["marking_points"]:
            # only infer T- points
            if mp["type"] != "T":
                continue

            # infer forward vertex
            bf = mp["branch_forward"]
            if not bf["paired"]:
                self.infer_branch_forward(bf)

            # infer right vertex
            br = mp["branch_right"]
            if not br["paired"] and self.infer_branch_right_left(br):
                sr = {}
                sr["idx"] = counter
                sr["p0x"] = mp["p0x"]
                sr["p0y"] = mp["p0y"]
                sr["p1x"] = int(bf["p1x"])
                sr["p1y"] = int(bf["p1y"])
                sr["p2x"] = int(br["p2x"])
                sr["p2y"] = int(br["p2y"])
                sr["p3x"] = int(br["p3x"])
                sr["p3y"] = int(br["p3y"])
                print("right slot[{}]: {}".format(counter, sr))
                counter += 1
                self.result_json["slots"].append(sr)

            # infer left vertex
            bl = mp["branch_left"]
            if not bl["paired"] and self.infer_branch_right_left(bl):
                sl = {}
                sl["idx"] = counter
                sl["p0x"] = mp["p0x"]
                sl["p0y"] = mp["p0y"]
                sl["p1x"] = int(bf["p1x"])
                sl["p1y"] = int(bf["p1y"])
                sl["p2x"] = int(bl["p2x"])
                sl["p2y"] = int(bl["p2y"])
                sl["p3x"] = int(bl["p3x"])
                sl["p3y"] = int(bl["p3y"])
                print("left slot[{}]: {}".format(counter, sl))
                counter += 1
                self.result_json["slots"].append(sl)
