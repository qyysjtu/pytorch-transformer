
import torch
from torch.profiler import ExecutionGraphObserver

out_file_prefix = "transformer"
# use_cuda = False
et = None

# 开启ET 
# if args.et:
et_file = f"{out_file_prefix}_et.json"
et = ExecutionGraphObserver()
et.register_callback(et_file)
et.start()

# 启用并配置Torch的Profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA  # 同时监控CPU和CUDA的活动
    ],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=1,
        active=3
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),  # 输出到logs目录
    record_shapes=True,
    profile_memory=True,  # 记录内存信息
    with_stack=True  # 记录堆栈信息
) as profiler:

    for idx in range(10):  # 运行一个批次的训练循环10次迭代
        et.start()
        # optimizer.zero_grad()
        # output = model(input)
        # loss = output.sum()
        # loss.backward()
        # optimizer.step()
        profiler.step()  # 更新profiler到下一步

# 关闭ET 
if et:
    et.stop()
    et.unregister_callback()
