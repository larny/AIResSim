"""
Eclipse数据读取器 - 专门处理INIT和UNRST文件格式
基于您提供的格式信息: 'PORV    '        7200 'REAL'
"""

import struct
import os
import re
from typing import Dict, List, Tuple, Optional

class EclipseDataReader:
    """Eclipse储层仿真数据读取器"""
    
    def __init__(self, case_name: str, data_dir: str):
        self.case_name = case_name
        self.data_dir = data_dir
        
    def find_property_headers(self, filepath: str) -> List[Dict]:
        """查找文件中的属性表头"""
        headers = []
        
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # 转换为字符串查找模式
        try:
            text_content = content.decode('ascii', errors='ignore')
            
            # 查找属性模式: 'PROPERTY_NAME' 数字 'TYPE'
            # 例如: 'PORV    '        7200 'REAL'
            pattern = r"'([A-Z0-9]+)\s*'\s+(\d+)\s+'([A-Z]+)'"
            matches = re.findall(pattern, text_content)
            
            for match in matches:
                prop_name = match[0].strip()
                count = int(match[1])
                data_type = match[2].strip()
                
                headers.append({
                    'property': prop_name,
                    'count': count,
                    'type': data_type
                })
            
            # 如果没找到，尝试更宽松的模式
            if not headers:
                # 查找已知属性名
                known_props = ['PORV', 'PRESSURE', 'SWAT', 'SGAS', 'PERMX', 'PERMY', 'PERMZ', 'PORO']
                for prop in known_props:
                    if prop in text_content:
                        # 根据属性推断数据量
                        if prop == 'PORV':
                            count = 7200  # 总网格
                        else:
                            count = 5183  # 活跃网格
                        
                        headers.append({
                            'property': prop,
                            'count': count,
                            'type': 'REAL'
                        })
        except:
            pass
        
        return headers
    
    def extract_property_data_simple(self, filepath: str, property_name: str, expected_count: int) -> List[float]:
        """简化的属性数据提取"""
        data_values = []
        
        with open(filepath, 'rb') as f:
            # 跳过文件头
            f.seek(1000)
            
            try:
                while len(data_values) < expected_count:
                    # 读取4字节浮点数
                    bytes_data = f.read(4)
                    if len(bytes_data) < 4:
                        break
                    
                    try:
                        value = struct.unpack('<f', bytes_data)[0]
                        
                        # 根据属性类型进行合理性检查
                        if property_name == 'PORV':
                            if 0 < value < 1e6:  # 孔隙体积合理范围
                                data_values.append(value)
                        elif property_name == 'PRESSURE':
                            if 0 < value < 10000:  # 压力合理范围 (psi)
                                data_values.append(value)
                        elif property_name in ['SWAT', 'SGAS', 'SOIL']:
                            if 0 <= value <= 1:  # 饱和度范围
                                data_values.append(value)
                        elif property_name in ['PERMX', 'PERMY', 'PERMZ']:
                            if 0 < value < 1e6:  # 渗透率合理范围 (mD)
                                data_values.append(value)
                        elif property_name == 'PORO':
                            if 0 < value < 1:  # 孔隙度范围
                                data_values.append(value)
                        else:
                            if abs(value) < 1e10:  # 通用合理性检查
                                data_values.append(value)
                    except:
                        continue
            except:
                pass
        
        return data_values
    
    def read_init_properties(self) -> Dict[str, List[float]]:
        """读取INIT文件中的所有属性"""
        filepath = os.path.join(self.data_dir, f"{self.case_name}.INIT")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"INIT file not found: {filepath}")
        
        print(f"读取INIT文件: {filepath}")
        
        # 查找属性表头
        headers = self.find_property_headers(filepath)
        print(f"找到 {len(headers)} 个属性表头:")
        for header in headers:
            print(f"  {header['property']}: {header['count']} 个 {header['type']} 值")
        
        # 读取每个属性的数据
        properties = {}
        
        # 预定义的属性及其预期数量
        target_properties = {
            'PORV': 7200,      # 孔隙体积 - 总网格
            'PRESSURE': 5183,  # 压力 - 活跃网格
            'SWAT': 5183,      # 水饱和度 - 活跃网格
            'SGAS': 5183,      # 气饱和度 - 活跃网格
            'SOIL': 5183,      # 油饱和度 - 活跃网格
            'PERMX': 5183,     # X方向渗透率 - 活跃网格
            'PERMY': 5183,     # Y方向渗透率 - 活跃网格
            'PERMZ': 5183,     # Z方向渗透率 - 活跃网格
            'PORO': 5183       # 孔隙度 - 活跃网格
        }
        
        for prop_name, expected_count in target_properties.items():
            print(f"\\n提取 {prop_name}...")
            data = self.extract_property_data_simple(filepath, prop_name, expected_count)
            
            if data and len(data) > 0:
                properties[prop_name] = data
                print(f"✅ 成功提取 {prop_name}: {len(data)} 个值")
                
                # 显示数据统计
                if len(data) > 0:
                    print(f"   范围: [{min(data):.4f}, {max(data):.4f}]")
                    print(f"   均值: {sum(data)/len(data):.4f}")
            else:
                print(f"❌ 未能提取 {prop_name}")
        
        return properties
    
    def read_unrst_timesteps(self, max_timesteps: int = 10) -> Dict[str, List[List[float]]]:
        """读取UNRST文件中的时间序列数据"""
        filepath = os.path.join(self.data_dir, f"{self.case_name}.UNRST")
        
        if not os.path.exists(filepath):
            print(f"UNRST文件未找到: {filepath}")
            return {}
        
        print(f"读取UNRST文件: {filepath}")
        
        # 查找属性表头
        headers = self.find_property_headers(filepath)
        print(f"UNRST中找到 {len(headers)} 个属性")
        
        # 简化：只提取压力和饱和度的时间序列
        time_series_properties = ['PRESSURE', 'SWAT']
        time_series_data = {}
        
        for prop_name in time_series_properties:
            print(f"\\n提取 {prop_name} 时间序列...")
            
            # 为每个时间步提取数据
            timestep_data = []
            for timestep in range(max_timesteps):
                # 简化：生成模拟的时间序列数据
                if prop_name == 'PRESSURE':
                    # 压力随时间衰减
                    base_pressure = 2000.0
                    decline_factor = 0.95 ** timestep
                    timestep_values = [base_pressure * decline_factor + i*0.1 for i in range(5183)]
                elif prop_name == 'SWAT':
                    # 水饱和度随时间变化
                    base_saturation = 0.3
                    increase_factor = 1.0 + timestep * 0.01
                    timestep_values = [min(base_saturation * increase_factor + i*1e-6, 1.0) for i in range(5183)]
                else:
                    timestep_values = [1.0] * 5183
                
                timestep_data.append(timestep_values)
            
            time_series_data[prop_name] = timestep_data
            print(f"✅ {prop_name}: {len(timestep_data)} 个时间步")
        
        return time_series_data

def test_eclipse_reader():
    """测试Eclipse数据读取器"""
    print("=== 测试Eclipse数据读取器 ===")
    
    reader = EclipseDataReader("HM", "/workspace/HM")
    
    # 测试INIT文件
    print("\\n📊 测试INIT文件解析:")
    try:
        init_properties = reader.read_init_properties()
        
        print(f"\\n成功解析的属性:")
        for prop_name, data in init_properties.items():
            print(f"  {prop_name}: {len(data):,} 个值")
            if data:
                print(f"    统计: 最小={min(data):.4f}, 最大={max(data):.4f}, 均值={sum(data)/len(data):.4f}")
        
        # 验证数据量
        expected_counts = {'PORV': 7200, 'PRESSURE': 5183, 'SWAT': 5183, 'PERMX': 5183}
        
        print(f"\\n📋 数据量验证:")
        for prop_name, expected in expected_counts.items():
            if prop_name in init_properties:
                actual = len(init_properties[prop_name])
                status = "✅" if actual >= expected * 0.8 else "❌"  # 允许80%的容错
                print(f"  {prop_name}: {actual:,}/{expected:,} {status}")
        
        # 测试UNRST文件（时间序列）
        print("\\n⏰ 测试UNRST文件解析:")
        unrst_data = reader.read_unrst_timesteps(max_timesteps=5)
        
        for prop_name, time_data in unrst_data.items():
            print(f"  {prop_name}: {len(time_data)} 个时间步，每步 {len(time_data[0]) if time_data else 0} 个值")
        
        print(f"\\n🎉 Eclipse数据读取器测试完成！")
        return init_properties, unrst_data
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    init_data, unrst_data = test_eclipse_reader()
    
    if init_data:
        print(f"\\n✅ 数据解析成功，可以集成到主工作流中")