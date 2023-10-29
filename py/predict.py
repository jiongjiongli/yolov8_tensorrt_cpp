

# self.setup_model(model)

import tensorrt as trt
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
logger = trt.Logger(trt.Logger.INFO)

with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
    meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
    metadata = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
    model = runtime.deserialize_cuda_engine(f.read())  # read engine
context = model.create_execution_context()
bindings = OrderedDict()
output_names = []
fp16 = False  # default updated below
dynamic = False
for i in range(model.num_bindings):
    name = model.get_binding_name(i)
    dtype = trt.nptype(model.get_binding_dtype(i))
    if model.binding_is_input(i):
        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
            dynamic = True
            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
        if dtype == np.float16:
            fp16 = True
    else:  # output
        output_names.append(name)
    shape = tuple(context.get_binding_shape(i))
    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size


# im = self.preprocess(im0s)
    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

# preds = self.inference(im, *args, **kwargs)
    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        # return self.model(im)
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)
        s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)


# self.results = self.postprocess(preds, im, im0s)
    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
