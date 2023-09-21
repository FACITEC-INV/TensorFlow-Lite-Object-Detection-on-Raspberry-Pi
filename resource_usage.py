import psutil

class ResourceUsage:
    def __init__(self):
        self.first_info = None
        self.info_array = []
    
    def addInfo(self):
        try:
            ram_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            if(self.first_info == None):
                self.first_info = [cpu_percent,ram_info.percent,(ram_info.used / 1024 / 1024 )]
            else:
                self.info_array.append([cpu_percent,ram_info.percent,(ram_info.used / 1024 / 1024 )])
            
        except FileNotFoundError:
            print("Info not available on this system")
    
    def printInfo(self):
        cpu_sum,ram_per_sum,ram_sum = 0,0,0
        for info in self.info_array:
            cpu_sum += info[0]
            ram_per_sum += info[1]
            ram_sum += info[2]
        
        print(f"Ram usage - prev: {round(self.first_info[2])} MB")
        print(f"Ram usage % - prec: {round(self.first_info[1],2)}%")
        print(f"CPU usage % - prev: {round(self.first_info[0],2)}%")
        
        print(f"Ram usage: {round(ram_sum/len(self.info_array))} MB")
        print(f"Ram usage %: {round(ram_per_sum/len(self.info_array),2)}%")
        print(f"CPU usage %: {round(cpu_sum/len(self.info_array),2)}%")
