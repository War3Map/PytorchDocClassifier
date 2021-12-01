import pandas
import os
from pathlib import Path
from classifier.classifier_settings import DS_PATH

markup = []
labels = []

for root, dirs, files in os.walk(DS_PATH):
    for file in files:
        file_path = os.path.join(root, file)
        label = Path(root).name
        markup.append(
            (file_path, label)
        )
        if label not in labels:
            labels.append(label)

print(f"Лейблы {labels}")

df = pandas.DataFrame(markup, columns=['path', 'label'])
df.to_csv("ds_markup.csv", encoding="utf-8", index=False)
