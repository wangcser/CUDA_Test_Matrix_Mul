import time as t


class timer:

    start_time = 0
    stop_time = 0
    tt = {}

    # 计时开始
    def start(self):
        self.start_time = t.time()
        print("计时开始", self.start_time)

    # 计时结束
    def stop(self):
        self.stop_time = t.time()
        self.__calc_lasted_time()
        print("计时结束", self.stop_time)

    # 计算持续时间，内部方法
    def __calc_lasted_time(self):
        self.lasted = round(self.stop_time - self.start_time, 6)
        self.prompt = "time cost: "
        self.prompt = self.prompt + str(self.lasted) + 's'
        print(self.prompt)


if __name__ == '__main__':

    timer = timer()
    timer.start()
    t.sleep(1)
    timer.stop()