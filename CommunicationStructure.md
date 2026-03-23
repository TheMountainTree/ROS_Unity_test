# Structure
##  Communication Between ROS Node and Unity
### Data Structure
#### Data Plane
ros -> Unity
1. Image
#### Control Plane
ros -> Unity
1. phase
    1. prepare
    2. stim
    3. stop
    4. done
2. trial_id/group_id
3. slot_states (merge `activate_slots` + `flash_slots`)
   1. `hidden`: do not render, no flash
   2. `visible_static`: render, no flash
   3. `visible_flash`: render and flash
4. slot_to_candidate_id (optional, for traceability)
5. session_id/seq_id
**Attention**: Unity component `SSVEP_Stimulus2` should parse dynamic `slot_states` instead of fixed 6-image assumptions.
#### Ack/Feedback Plane
Unity -> ros
1. `entered_prepare` (Unity entered prepare state)
2. `entered_stim` (Unity entered stim state; equivalent to current `trial_started/stim_started`)
3. `entered_stop` (Unity stopped stimulation/render loop)
4. `entered_done` (Unity entered done/final state)
5. `error` (failed to enter requested state, with reason)
6. ack carries `session_id/seq_id`
#### Handshake Plane
1. ready
2. heartbeat
3. timeout

## Communication Between ROS Node and Reasoner
### Data Structure
#### Data Plane
Reasoner -> ros
1. `/reasoner/candidates`
    1. a batch with N(1..6) candidates
2. round_id
3. group_id
4. constraints
    1. `round_id` must be strictly increasing within one run
    2. rollback should create a new `round_id` (do not reuse old round id)
    3. `group_id` can repeat (for rollback/resend)
#### Control Plane
ros <-> reasoner
`/reasoner/control`
`/reasoner/feedback`
1. phase
    1. request_batch
    2. selection
    3. rollback
    4. confirm
    5. done
    6. cancel (optional)
`round_id + group_id + timestamp`
#### Status Feedback Plane
reasoner <-> ros
`/reasoner/status`
`/reasoner/error`
1. idle
2. waiting_batch
3. batch_ready
4. waiting_batch
5. finished
6. timeout
7. failed
`code + message + recoverable + round_id + group_id`
#### Handshake Plane
1. ready
2. version
3. heartbeat
4. timeout

场景构建
# Pretrain
1. 预训练进行3（每个槽位3次）*8（槽位）轮（可自定义次数）
2. 生成7个白色图片和1个红色图片（红色图片用于提示用户看哪个）
3. 然后随机一个槽位为红色图片，剩下的为白色图片
4. ROS2 发送给Unity命令`prepare`
5. ROS2 发送给Unity,让Unity展示，提示用户
6. ROS2 计时满足cue的时间后发送命令`stim`，Unity接受到开始闪烁，同时ROS2也要给trigger发送`1`，用于给eeg信号打标。同时维护一个buffer接受数据，从第9通道接受到`1`后开始写入数据。
7. ROS2 计时满足stim的时间后发送命令`stop`，Unity接受到后停止闪烁，同时ROS2也要给trigger发送`2`，用于给eeg信号打标。当第9通道接受到`2`后，截止数据，开始把`1`-`2`之间的数据写入。并命名为`{round_id}_{index_id}`。{index_id}是指随机的槽位的唯一index
8. ROS2 计时满足rest时间后，发送`prepare`命令，进入4的阶段
9. 当总次数到达3*8时，发送`done`命令，结束训练。

# Decode
1. reasoner首先会发送`candidates`和`active_slots`给ros
2. ros先发送命令`prepare`。ros会根据`candidates`和`active_slots`来把对应的图片发送给Unity展示出来，并且根据`active_slots`来决定哪些槽位可用，可用的会闪烁，不可用的不会闪烁。Unity进入`prepare`状态，并展示让用户看清楚选项。
3. ros等待prepare的时间结束，发送命令`stim`，Unity接受到开始闪烁，同时ROS2也要给trigger发送`1`，用于给eeg信号打标。同时维护一个buffer接受数据，从第9通道接受到`1`后开始写入数据。
4. ros计时满足stim的时间后，发送命令`stop`，Unity接受到后停止闪烁，同时ROS2也要给trigger发送`2`，用于给eeg信号打标。当第9通道接受到`2`后，截止数据，开始把`1`-`2`之间的数据写入。并命名为`{round_id}`。
5. unity会继续展示选项，等待下一轮
6. 假设解码了结果，reasoner端等待ros解码完成，这里还没有实现，因此预留接口，现在先使用参数设置实现选择。
7. ros会根据解码结果，发送`selection`命令给reasoner,reasoner会根据命令决定下一批发送什么图片。