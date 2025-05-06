# post_data_process

统一格式
cot data/COT/qihoo360/Light-R1-SFTData推理和答案在一起  answer 是空

HelpSteer3 
-3：回复 1 比回复 2 好得多
-2：响应 1 优于响应 2
-1：响应 1 比响应 2 稍好
0：响应 1 与响应 2 大致相同
1：回复 2 比回复 1 略好
2：回复 2 优于回复 1
3：回复 2 比回复 1 好得多
如果是0，则不选这条数据


lmarena-ai 中如果chosen和rejected都为空，则平局

webdev-arena-preference-10k 和 arena-human-preference-100k 包含多轮对话，需要处理

OpenR1-Math-220k 如果chosen和rejected都为空，可能是平局也可能是少于两条推理或4条推理的平局



