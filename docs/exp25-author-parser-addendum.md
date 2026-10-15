# EXP25原作者解析补充说明

在6站pilot调用结束后、其余42站模型调用前登记。原冻结推理不改，严格解析记录完整保留。
pilot发现HTTP 200响应常含解释文字，原适配的json.loads无法读取；锁定作者源码
`result_agg.py::parse_output`原本包含方括号提取，且utils包含原向量转换函数。

使用已有`scripts/replay_author_parser.py::load_author_functions`加载未改作者函数，
对同一批已保存响应离线解析，增列`visualtime_author_parser`，0新增请求。
不根据标签选择解析结果、不改提示、不调阈值、不改作者边界截断行为。
六子请求有失败时仍保成功并集，另报整站零检测敏感性；严格解析结果单独报告。
此为登记后的评分接口补充，不称完整论文复现，也不替代冻结主方案。
