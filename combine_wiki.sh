#!/bin/bash

# 出力ファイル名
OUTPUT_FILE="combined_wiki_content.txt"

# スクリプト実行前に、古い出力ファイルがあれば削除する
rm -f "$OUTPUT_FILE"

# 最初にディレクトリ構造をtree形式で出力ファイルに書き込む
echo "--- Directory Tree ---" >> "$OUTPUT_FILE"
tree >> "$OUTPUT_FILE"
echo -e "\n--- End of Directory Tree ---\n" >> "$OUTPUT_FILE"

# findコマンドでカレントディレクトリとサブディレクトリ内の.mdファイルを再帰的に検索し、
# 1つずつループ処理を行う
# -print0 と read -r -d $'\0' の組み合わせは、ファイル名にスペースや特殊文字が含まれていても安全に処理するための定石
find . -name "*.md" -print0 | while IFS= read -r -d $'\0' file; do
  # ファイルパスをヘッダーとして出力
  echo "--- Start of file: $file ---" >> "$OUTPUT_FILE"
  # ファイルの内容を追記
  cat "$file" >> "$OUTPUT_FILE"
  # ファイルの終わりを示すフッターを出力
  echo -e "\n--- End of file: $file ---\n" >> "$OUTPUT_FILE"
done

echo "Wiki content has been successfully combined into $OUTPUT_FILE" 