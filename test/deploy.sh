echo "Getting multiclass predictions..."
curl -X 'POST' \
        'http://localhost/predict_multiclass_example' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
          "table_name": [
            "example_data", "example_data", "example_data", "example_data", "example_data"
          ],
          "id": [
            "22", "24", "19", "15", "18"
          ]
        }' | tee multiclass_predictions.txt

if ! grep -i -e "error" -e "warning" multiclass_predictions.txt; then
        rm multiclass_predictions.txt
        echo "\n☑ Multiclass prediction passed."
else
        rm multiclass_predictions.txt
        echo "\nMulticlass prediction failed."
        exit 1
fi

echo "Getting multilabel predictions..."
curl -X 'POST' \
        'http://localhost/predict_multilabel_example' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
          "table_name": [
            "example_data", "example_data", "example_data", "example_data", "example_data"
          ],
          "id": [
            "03", "05", "12", "06", "08"
          ]
        }' | tee multilabel_predictions.txt

if ! grep -i -e "error" -e "warning" multilabel_predictions.txt; then
        rm multilabel_predictions.txt
        echo "\n☑ Multilabel prediction passed."
else
        rm multilabel_predictions.txt
        echo "\nMultilabel prediction failed."
        exit 1
fi