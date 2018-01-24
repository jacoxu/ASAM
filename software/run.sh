echo "Start to run the main function ..."
# nohup python -u main_run.py > ASAM_nohup.out 2>&1 &
python main_run.py > /dev/null&
touch $!".pid"