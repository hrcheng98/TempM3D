import numpy as np
import cv2 as cv

class Vis3d():
    def __init__(self, range_lr, range_fb, resolution):
        self.l_b = range_lr[0]
        self.r_b = range_lr[1]
        self.b_b = range_fb[0]
        self.f_b = range_fb[1]
        self.resolution = resolution
        self.reset()

    def reset(self):
        self.map = np.zeros((int((self.f_b-self.b_b)*self.resolution),
                             int((self.r_b-self.l_b)*self.resolution),
                             3))

    def get_map(self):
        return self.map

    def _roty2d(self, t):
        ''' Rotation about the y-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, s],
                         [-s, c]])

    def add_bev_box(self, bev_center, wl, ry, color, thk=5):
        w, l = wl
        corners = np.array([[-w/2, -w/2, w/2, w/2],
                            [-l/2, l/2, l/2, -l/2]])
        R = self._roty2d(-ry)
        rot_corners = np.dot(R, corners).T + bev_center
        img_corners = self._convert_real2img(rot_corners)
        for i in range(4):
            self._draw_line(img_corners[i], img_corners[(i+1)%4], color, thk)

        if ry+np.pi/2 < 0:
            # self._draw_line(img_corners[1], img_corners[2], color, thk=40)
            self._draw_line(img_corners[1], img_corners[2], color, thk=thk)
        else:
            # self._draw_line(img_corners[3], img_corners[0], color, thk=40)
            self._draw_line(img_corners[3], img_corners[0], color, thk=thk)

    def add_points(self, points, color, thk=5):
        img_points = self._convert_real2img(points)
        for i in img_points:
            self._draw_circle(i, color, thk)

    def add_line(self, points, color, thk=5):
        img_points = self._convert_real2img(points)
        p1, p2 = img_points[0], img_points[1]
        self.map = cv.line(self.map, (p1[0], p1[1]), (p2[0], p2[1]), color, thk)

    def add_circle(self, points, color, R, thk=5):
        img_points = self._convert_real2img(points)
        img_R = int(self.resolution * R)
        for i in img_points:
            self.map = cv.circle(self.map, (i[0], i[1]), img_R, color, thk)

    def _convert_real2img(self, points):
        img_x = (points[:, 0] - self.l_b) * self.resolution
        img_y = (self.f_b-self.b_b)*self.resolution - \
                (points[:, -1] - self.b_b) * self.resolution
        return np.stack([img_x.astype(np.uint16), img_y.astype(np.uint16)], axis=1)

    def _draw_line(self, p1, p2, color, thk):
        self.map = cv.line(self.map, (p1[0], p1[1]), (p2[0], p2[1]), color, thk)

    def _draw_circle(self, p1, color, thk):
        self.map = cv.circle(self.map, (p1[0], p1[1]), thk, color, thk)

    def _draw_points(self, points, color, thk):
        self.map[points[:, 1], points[:, 0]] = color



def draw_pred_lidar(pred, pred2, R, lidar_points, mode='sphere'):
    pred_np = pred.detach().cpu().numpy()
    pred2_np = pred2.detach().cpu().numpy()
    lidar_points_np = lidar_points.cpu().numpy()
    visBox = Vis3d([-30, 30], [0, 100], 100)
    visBox.add_points(lidar_points_np[:, [0, 2]], (255, 255, 0), thk=2)

    if mode =='sphere':
        visBox.add_circle(np.array([pred_np[[0, 2]]]), (0, 255, 0), R, thk=2)
        visBox.add_circle(np.array([pred2_np[[0, 2]]]), (0, 0, 255), R, thk=2)
    else:
        # visBox.add_line(pred_np[:2], (255, 255, 255), thk=2)
        # visBox.add_line(pred_np[1:3], (255, 255, 255), thk=2)
        # visBox.add_line(pred_np[2:], (255, 255, 255), thk=2)
        # visBox.add_line(pred_np[[3, 0]], (255, 255, 255), thk=2)
        # visBox.add_bev_box(bev_center, wl, ry, (255, 255, 255), thk=2)
        visBox.add_bev_box(pred_np[:2], pred_np[2:4], pred_np[4], (255, 255, 255), thk=2)

    map = visBox.get_map()
    # map2 = np.concatenate([map[..., 0:1], map[..., 1:2], map[..., 2:3], map[..., 0:1]], axis=2)
    cv.imwrite('tmp.png', map)