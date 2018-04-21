import time as t


class timer:
    """
    class:  定义时间，用于计时
    param: 开始时间，结束时间
    func : __init__ 初始化时间类
           start 开始计时
           stop 结束计时
           __calc_lasted_time 计算持续时间

    """
    def __init__(self):
        self.start_time = 0
        self.stop_time = 0
        self.lasted = 0

    # 计时开始
    def start(self):
        self.start_time = t.time()

    # 计时结束
    def stop(self):
        self.stop_time = t.time()
        return self.__calc_lasted_time()

    # 计算持续时间，内部方法
    def __calc_lasted_time(self):
        self.lasted = round(self.stop_time - self.start_time, 6)
        self.prompt = "time cost: "
        self.prompt = self.prompt + str(self.lasted) + 's'
        # print(self.prompt)
        return self.lasted


if __name__ == '__main__':

    timer = timer()
    timer.start()
    t.sleep(1)
    timer.stop()