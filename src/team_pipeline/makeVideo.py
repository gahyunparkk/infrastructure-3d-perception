import os, cv2

extract_dir = "C:\\CES\\2dot5D\\samples\\RC"

output_path  = "C:\\CES\\2dot5D\\results\\RC.mp4"

# 4. PNG 파일 정렬
image_files = sorted([
    os.path.join(extract_dir, f)
    for f in os.listdir(extract_dir)
    if f.lower().endswith('.png')
])

# 5. 영상 생성
if not image_files:
    print(f"Error: No PNG files found in {extract_dir}")
else:
    fps = 10  # 원하는 FPS 값 
    
    # 타겟 크기
    target_w, target_h = 1920,1080

    out = cv2.VideoWriter(
        output_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        fps, 
        (target_w, target_h)
    )

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to load: {img_path}")
            continue
        
        # 리사이즈
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        out.write(img)

    out.release()
    print(f"🎉 완성된 영상: {output_path}")
