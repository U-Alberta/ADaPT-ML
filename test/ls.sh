echo "Try sampling some tasks..."
docker exec label-studio-dev python ./ls/sample_tasks.py example_data txt 30 example --filename example_tasks.json
docker exec label-studio-dev sh -c "! grep -i -e "error" -e "warning" /tasks/*log.txt"
echo "☑ Sampling tasks passed."

echo "Try processing annotations with random gold choice..."
docker exec label-studio-dev python ./ls/process_annotations.py example_annotations.json example random
docker exec label-studio-dev sh -c "! grep -i -e "error" -e "warning" /annotations/example/process_log.txt"
echo "☑ Random gold choice passed."

echo "Try processing annotations with drop gold choice..."
docker exec label-studio-dev python ./ls/process_annotations.py example_annotations.json example drop
docker exec label-studio-dev sh -c "! grep -i -e "error" -e "warning" /annotations/example/process_log.txt"
echo "☑ Drop gold choice passed."

echo "Try processing annotations with majority gold choice..."
docker exec label-studio-dev python ./ls/process_annotations.py example_annotations.json example majority
docker exec label-studio-dev sh -c "! grep -i -e "error" -e "warning" /annotations/example/process_log.txt"
echo "☑ Majority gold choice passed."

echo "Try processing annotations with worker_1 gold choice..."
docker exec label-studio-dev python ./ls/process_annotations.py example_annotations.json example 1
docker exec label-studio-dev sh -c "! grep -i -e "error" -e "warning" /annotations/example/process_log.txt"
echo "☑ Worker_1 gold choice passed."

echo "Check annotator agreement..."
docker exec label-studio-dev python ./ls/annotator_agreement.py example
docker exec label-studio-dev sh -c "! grep -iq -e "error" -e "warning" /annotations/example/agreement_log.txt"
docker exec label-studio-dev sh -c "grep -iq "0.43847133757961776" /annotations/example/agreement_log.txt"
echo "☑ Annotator agreement check passed."
