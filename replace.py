def replace_path(input_file, output_file):
    # 读取输入文件内容
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 替换路径
    modified_lines = []
    for line in lines:
        # 使用replace方法进行路径替换
        modified_line = line.replace('datasets/hrsid/image/', 'H:/yk/dataset/HRSID/HRSID_JPG/JPEGImages/')
        modified_lines.append(modified_line)

    # 将修改后的内容写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(modified_lines)


# 输入和输出文件路径
input_file = r'H:\yk\dataset\HRSID\HRSID_JPG\flist\train_inshore_2.txt'  # 输入文件路径，请根据实际情况修改
output_file = r'H:\yk\dataset\HRSID\HRSID_JPG\flist\train_inshore_2_new.txt'  # 输出文件路径，请根据实际情况修改

# 执行路径替换
replace_path(input_file, output_file)
print(f"路径替换完成，已保存到 {output_file}")