import cv2
import easyocr
import pandas as pd
import csv
# 初始化 OCR 识别器（中英皆可）
reader = easyocr.Reader(['en'], gpu=False)

def extract_label_and_value(img):
    # 读取图像
    # img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 可选：增强对比度
    # _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # OCR 识别
    results = reader.readtext(gray)

    extracted = {}
    for (bbox, text, conf) in results:
        text = text.upper().strip()
        if text in ['PV', 'SV']:
            # 获取其右边区域的值
            x_min = int(bbox[1][0])
            x_max = int(bbox[2][0])
            y_min = int(bbox[1][1])
            y_max = int(bbox[2][1])

            # 在原图上向右扩展一个区域
            value_region = img[y_min-10:y_max+10, x_max:x_max+100]
            value_results = reader.readtext(value_region)
            if value_results:
                extracted[text] = value_results[0][1]
    
    return extracted

def preprocess_panel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 中值滤波去噪
    blurred = cv2.medianBlur(gray, 3)
    # 增强对比 + 自适应阈值
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=3
    )
    return thresh


# 示例调用
if __name__ == '__main__':
    df = pd.read_csv('./contour_img/part6.csv', quotechar='"')
    for _, row in df.iterrows():
        image_name = row[0]
        img = cv2.imread(f'./contour_img/{image_name}')
        result_str = row[1]
        panels = result_str.split('panel')
        data = []
        # process one panel
        for i, p in enumerate(panels):
            if not p.strip():
                continue
            score_start = p.find('(')
            score_end = p.find(')')
            score = float(p[score_start+1: score_end])
            box_start = p.find('[')
            box_end = p.find(']')
            box = [float(x) for x in p[box_start+1: box_end].split()]
            x1, y1, x2, y2 = map(int, box)
            sub_img = img[y1:y2, x1:x2]
            sub_img = preprocess_panel(sub_img)
            cv2.imwrite(f'{i}.jpg', sub_img)
        #     results = reader.readtext(sub_img)
        #     for bbox, text, conf in results:
        #         data.append({
        #             'panel_idx': i,
        #             'text': text,
        #             'confidence': round(conf, 4),
        #         })
        # df = pd.DataFrame(data)
        # df.to_csv(f'./output/{image_name}.csv', index=False, encoding='utf-8-sig')
        break
            # result = extract_label_and_value(sub_img)
            # print(result)  # 例如 {'PV': '156', 'SV': '200'}
