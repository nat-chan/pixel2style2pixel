while curl -s https://gradio.app -o /dev/null
do
    start=$(date "+%s")
#    sleep 3
    python script_demo_master.py
    end=$(date "+%s")
    t=$((end-start))
    if [ $t -lt 60 ]; then
        echo $t sec
        break
    fi
done