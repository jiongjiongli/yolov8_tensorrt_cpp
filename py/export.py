
f = self.file.with_suffix('.engine')

logger = trt.Logger(trt.Logger.INFO)

builder = trt.Builder(logger)
config = builder.create_builder_config()
config.max_workspace_size = self.args.workspace * 1 << 30
flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f'failed to load ONNX file: {f_onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

if builder.platform_has_fast_fp16 and self.args.half:
    config.set_flag(trt.BuilderFlag.FP16)

del self.model
torch.cuda.empty_cache()

# Write file
with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
    # Metadata
    meta = json.dumps(self.metadata)
    t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
    t.write(meta.encode())
    # Model
    t.write(engine.serialize())
