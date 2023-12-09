for file in $(ls $1)
do
  echo "Running evaluation on" $file
   python evals/compute_pfid.py --gt-folder $2  --pred-folder  $1/$file/exam_generate_dir --workers 60
done