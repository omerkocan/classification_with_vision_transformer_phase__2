import os
import shutil
import random


DATA_DIR = r"C:/Users/pc/Downloads/UCMerced/UCMerced_LandUse/Images"
OUTPUT_DIR = r"C:/Users/pc/Downloads/UCMerced"
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# create new folders
train_dir = os.path.join(OUTPUT_DIR, "UC_Merced_Train")
test_dir = os.path.join(OUTPUT_DIR, "UC_Merced_Test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(DATA_DIR):
    class_src = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_src):
        continue

    class_train = os.path.join(train_dir, class_name)
    class_test = os.path.join(test_dir, class_name)
    os.makedirs(class_train, exist_ok=True)
    os.makedirs(class_test, exist_ok=True)

    images = [f for f in os.listdir(class_src)
              if os.path.isfile(os.path.join(class_src, f))]
    random.shuffle(images)
    n_train = int(len(images) * TRAIN_RATIO)

    train_imgs = images[:n_train]
    test_imgs = images[n_train:]

    for fname in train_imgs:
        shutil.copy2(os.path.join(class_src, fname),
                     os.path.join(class_train, fname))
    for fname in test_imgs:
        shutil.copy2(os.path.join(class_src, fname),
                     os.path.join(class_test, fname))

    print(f"{class_name}: {len(train_imgs)} train, {len(test_imgs)} test")

print("'UC_Merced_Train' ve 'UC_Merced_Test' klasörleri oluşturuldu.")
