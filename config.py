class Config():
    def __init__(self) -> None:
        self.image_root = '/home/pz1/datasets/sod/images/DUTS_class'
        self.gt_root = '/home/pz1/datasets/sod/gts/DUTS_class'

        # self-supervision
        self.lambda_loss_ss = 0.3   # 0 means no self-supervision

        # label smoothing
        self.label_smooth = 0.001   # epsilon for smoothing, 0 means no label smoothing, 

        # preproc
        self.preproc_activated = True
        self.hflip_prob = 0.5
        self.crop_border = 30      # < 1 as percent of min(wid, hei), >=1 as pixel
        self.rotate_prob = 0.2
        self.rotate_angle = 15
        self.enhance_activated = True
        self.enhance_brightness = (5, 15)
        self.enhance_contrast = (5, 15)
        self.enhance_color = (0, 20)
        self.enhance_sharpness = (0, 30)
        self.gaussian_mean = 0.1
        self.gaussian_sigma = 0.35
        self.pepper_noise = 0.0015
        self.pepper_turn = 0.5
