import csv
import math
import os
import socket
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

from psychopy import data, event, monitors
import numpy as np
from metabci.brainstim.paradigm import (
    SSVEP,
    P300,
    MI,
    AVEP,
    SSAVEP,
    paradigm,
    pix2height,
    code_sequence_generate,
)
from metabci.brainstim.framework import Experiment
from metabci.brainstim.utils import NeuroScanPort, NeuraclePort
from psychopy.tools.monitorunittools import deg2pix

try:
    from .utils import CircularEEGBuffer
except ImportError:
    from utils import CircularEEGBuffer


class Node4StyleEEGRecorder:
    """Node4_test-style EEG recorder with trigger(1/2) epoch alignment."""

    def __init__(
        self,
        save_dir="data/metabci_stim",
        eeg_server_ip="192.168.56.3",
        eeg_server_port=8712,
        eeg_n_channels=8,
        eeg_frame_floats=9,
        eeg_fs=1000.0,
        eeg_recv_buffer_size=4096,
        trigger_local_ip="192.168.56.103",
        trigger_local_port=5006,
        trigger_remote_ip="192.168.56.3",
        trigger_remote_port=8888,
        ring_buffer_seconds=20.0,
    ):
        self.save_dir = str(Path(save_dir).resolve())
        os.makedirs(self.save_dir, exist_ok=True)

        self.eeg_server_ip = eeg_server_ip
        self.eeg_server_port = int(eeg_server_port)
        self.eeg_n_channels = int(eeg_n_channels)
        self.eeg_frame_floats = int(eeg_frame_floats)
        self.eeg_fs = float(eeg_fs)
        self.eeg_recv_buffer_size = int(eeg_recv_buffer_size)
        self.eeg_frame_bytes = self.eeg_frame_floats * 4
        self.eeg_unpack_fmt = f"<{self.eeg_frame_floats}f"

        self.trigger_local_ip = trigger_local_ip
        self.trigger_local_port = int(trigger_local_port)
        self.trigger_remote_ip = trigger_remote_ip
        self.trigger_remote_port = int(trigger_remote_port)

        self.eeg_ring = CircularEEGBuffer(
            n_channels=self.eeg_n_channels,
            fs=self.eeg_fs,
            buffer_seconds=max(10.0, float(ring_buffer_seconds)),
        )
        self.eeg_tcp_sock = None
        self.eeg_tcp_buffer = bytearray()
        self.eeg_connected = False
        self.eeg_reconnect_at = 0.0
        self.stop_event = threading.Event()
        self.thread = None

        self.trigger_send_sock = self._create_trigger_sender()

        self.dataset_x = []
        self.dataset_y = []
        self.last_trigger_value = 0
        self.current_trial = None
        self._lock = threading.Lock()
        self.dataset_saved = False

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_stamp = run_stamp
        self.trials_csv_path = os.path.join(self.save_dir, f"ssvep4_pretrain_trials_{run_stamp}.csv")
        self.meta_csv_path = os.path.join(self.save_dir, f"ssvep4_pretrain_metadata_{run_stamp}.csv")
        self.dataset_path = os.path.join(self.save_dir, f"ssvep4_pretrain_dataset_{run_stamp}.npy")
        self.trials_csv_file = open(self.trials_csv_path, "w", newline="", encoding="utf-8")
        self.meta_csv_file = open(self.meta_csv_path, "w", newline="", encoding="utf-8")
        self.trials_writer = csv.writer(self.trials_csv_file)
        self.meta_writer = csv.writer(self.meta_csv_file)
        self.trials_writer.writerow(
            [
                "trial_id",
                "target_id",
                "target_frequency_hz",
                "cue_start_wall",
                "stim_start_wall",
                "stim_end_wall",
                "trigger_received",
                "trigger_wall",
                "stim_start_trigger_sent",
                "stim_start_trigger_wall",
                "stim_end_trigger_sent",
                "stim_end_trigger_wall",
                "stim_enter_abs",
                "stim_exit_abs",
                "epoch_start_abs",
                "epoch_end_abs_inclusive",
                "raw_samples",
                "epoch_complete",
                "epoch_saved",
            ]
        )
        self.meta_writer.writerow(
            [
                "trial_id",
                "target_id",
                "label",
                "stim_start_wall",
                "stim_end_wall",
                "epoch_start_abs",
                "epoch_end_abs_inclusive",
                "n_samples",
                "epoch_complete",
            ]
        )
        self.trials_csv_file.flush()
        self.meta_csv_file.flush()

    def _create_trigger_sender(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.trigger_local_ip, self.trigger_local_port))
        except OSError as exc:
            print(
                f"[metabci_stim] trigger bind failed on "
                f"{self.trigger_local_ip}:{self.trigger_local_port}: {exc}. fallback to auto-bind."
            )
        sock.connect((self.trigger_remote_ip, self.trigger_remote_port))
        return sock

    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _connect_eeg(self):
        now = time.monotonic()
        if self.eeg_connected or now < self.eeg_reconnect_at:
            return
        if self.eeg_tcp_sock is not None:
            try:
                self.eeg_tcp_sock.close()
            except Exception:
                pass
            self.eeg_tcp_sock = None
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((self.eeg_server_ip, self.eeg_server_port))
            sock.settimeout(0.05)
            self.eeg_tcp_sock = sock
            self.eeg_connected = True
            print(f"[metabci_stim] EEG TCP connected: {self.eeg_server_ip}:{self.eeg_server_port}")
        except Exception as exc:
            try:
                sock.close()
            except Exception:
                pass
            self.eeg_tcp_sock = None
            self.eeg_connected = False
            self.eeg_reconnect_at = now + 1.0
            print(f"[metabci_stim] EEG TCP connect failed: {exc}; retry in 1s")

    def _reader_loop(self):
        while not self.stop_event.is_set():
            self._connect_eeg()
            if not self.eeg_connected or self.eeg_tcp_sock is None:
                time.sleep(0.05)
                continue
            try:
                chunk = self.eeg_tcp_sock.recv(self.eeg_recv_buffer_size)
                if not chunk:
                    raise ConnectionError("server closed connection")
                self.eeg_tcp_buffer.extend(chunk)
            except socket.timeout:
                continue
            except Exception:
                self.eeg_connected = False
                self.eeg_reconnect_at = time.monotonic() + 1.0
                if self.eeg_tcp_sock is not None:
                    try:
                        self.eeg_tcp_sock.close()
                    except Exception:
                        pass
                    self.eeg_tcp_sock = None
                continue
            self._consume_tcp_buffer()

    def _consume_tcp_buffer(self):
        n_frames = len(self.eeg_tcp_buffer) // self.eeg_frame_bytes
        if n_frames <= 0:
            return
        eeg_chunk = np.empty((self.eeg_n_channels, n_frames), dtype=np.float32)
        trigger_values = []
        for i in range(n_frames):
            start = i * self.eeg_frame_bytes
            end = start + self.eeg_frame_bytes
            vals = struct.unpack(self.eeg_unpack_fmt, self.eeg_tcp_buffer[start:end])
            eeg_chunk[:, i] = vals[: self.eeg_n_channels]
            trigger_values.append(int(round(vals[self.eeg_n_channels])))
        del self.eeg_tcp_buffer[: n_frames * self.eeg_frame_bytes]
        start_abs, _ = self.eeg_ring.append(eeg_chunk)
        for i, trigger_value in enumerate(trigger_values):
            self._process_trigger_sample(start_abs + i, trigger_value)

    def _send_trigger(self, value):
        wall = datetime.now().isoformat(timespec="milliseconds")
        try:
            payload = int(value).to_bytes(1, byteorder="little", signed=False)
            self.trigger_send_sock.send(payload)
            return True, wall
        except Exception as exc:
            print(f"[metabci_stim] send trigger={value} failed: {exc}")
            return False, wall

    def start_trial(self, trial_id, target_id, target_freq_hz):
        with self._lock:
            self.current_trial = {
                "trial_id": int(trial_id),
                "target_id": int(target_id),
                "target_frequency_hz": float(target_freq_hz),
                "cue_start_wall": datetime.now().isoformat(timespec="milliseconds"),
                "stim_start_wall": "",
                "stim_end_wall": "",
                "trigger_received": False,
                "trigger_wall": "",
                "stim_start_trigger_sent": False,
                "stim_start_trigger_wall": "",
                "stim_end_trigger_sent": False,
                "stim_end_trigger_wall": "",
                "stim_enter_abs": -1,
                "stim_exit_abs": -1,
                "stim_start_abs": -1,
                "stim_end_abs_inclusive": -1,
                "raw_samples": 0,
                "epoch_complete": False,
                "epoch_saved": False,
                "epoch_start_pending": False,
            }

    def on_stim_start(self):
        with self._lock:
            if self.current_trial is None:
                return
            self.current_trial["stim_start_wall"] = datetime.now().isoformat(timespec="milliseconds")
            self.current_trial["stim_enter_abs"] = self.eeg_ring.latest_abs_index
            sent, wall = self._send_trigger(1)
            self.current_trial["stim_start_trigger_sent"] = bool(sent)
            self.current_trial["stim_start_trigger_wall"] = wall

    def on_stim_end(self):
        with self._lock:
            if self.current_trial is None:
                return
            self.current_trial["stim_end_wall"] = datetime.now().isoformat(timespec="milliseconds")
            self.current_trial["stim_exit_abs"] = self.eeg_ring.latest_abs_index
            sent, wall = self._send_trigger(2)
            self.current_trial["stim_end_trigger_sent"] = bool(sent)
            self.current_trial["stim_end_trigger_wall"] = wall

    def _process_trigger_sample(self, abs_index, trigger_value):
        if trigger_value == self.last_trigger_value:
            return
        self.last_trigger_value = trigger_value
        with self._lock:
            if self.current_trial is None:
                return
            if trigger_value == 1:
                self.current_trial["trigger_received"] = True
                self.current_trial["trigger_wall"] = datetime.now().isoformat(timespec="milliseconds")
                if not self.current_trial["epoch_complete"] and not self.current_trial["epoch_start_pending"]:
                    self.current_trial["stim_start_abs"] = int(abs_index)
                    self.current_trial["epoch_start_pending"] = True
                return
            if trigger_value == 2:
                if self.current_trial["epoch_start_pending"] and not self.current_trial["epoch_complete"]:
                    self.current_trial["stim_end_abs_inclusive"] = int(abs_index)
                    self.current_trial["epoch_complete"] = True
                    self.current_trial["epoch_start_pending"] = False
                    self._capture_current_epoch_locked()

    def _capture_current_epoch_locked(self):
        trial = self.current_trial
        if trial is None:
            return
        start_abs = int(trial["stim_start_abs"])
        end_abs_inclusive = int(trial["stim_end_abs_inclusive"])
        if start_abs < 0 or end_abs_inclusive < start_abs:
            return
        end_exclusive = end_abs_inclusive + 1
        if not self.eeg_ring.has_range(start_abs, end_exclusive):
            return
        try:
            epoch = self.eeg_ring.get_range(start_abs, end_exclusive)
        except Exception:
            return
        trial["raw_samples"] = int(epoch.shape[1])
        trial["epoch_saved"] = True
        self.dataset_x.append(epoch.astype(np.float32))
        self.dataset_y.append(int(trial["target_id"]))
        self.meta_writer.writerow(
            [
                trial["trial_id"],
                trial["target_id"],
                trial["target_id"],
                trial["stim_start_wall"],
                trial["stim_end_wall"],
                trial["stim_start_abs"],
                trial["stim_end_abs_inclusive"],
                trial["raw_samples"],
                int(trial["epoch_complete"]),
            ]
        )
        self.meta_csv_file.flush()
        self._save_dataset_locked()

    def _save_dataset_locked(self):
        if not self.dataset_x:
            return
        x_data = np.empty(len(self.dataset_x), dtype=object)
        for i, epoch in enumerate(self.dataset_x):
            x_data[i] = epoch
        y_data = np.asarray(self.dataset_y, dtype=np.int32)
        np.save(self.dataset_path, {"x": x_data, "y": y_data}, allow_pickle=True)
        self.dataset_saved = True

    def finalize_trial(self):
        with self._lock:
            if self.current_trial is None:
                return
            t = self.current_trial
            self.trials_writer.writerow(
                [
                    t["trial_id"],
                    t["target_id"],
                    f"{t['target_frequency_hz']:.3f}",
                    t["cue_start_wall"],
                    t["stim_start_wall"],
                    t["stim_end_wall"],
                    int(t["trigger_received"]),
                    t["trigger_wall"],
                    int(t["stim_start_trigger_sent"]),
                    t["stim_start_trigger_wall"],
                    int(t["stim_end_trigger_sent"]),
                    t["stim_end_trigger_wall"],
                    t["stim_enter_abs"],
                    t["stim_exit_abs"],
                    t["stim_start_abs"],
                    t["stim_end_abs_inclusive"],
                    t["raw_samples"],
                    int(t["epoch_complete"]),
                    int(t["epoch_saved"]),
                ]
            )
            self.trials_csv_file.flush()
            self.current_trial = None

    def wait_current_trial_epoch(self, timeout_s=0.8, poll_s=0.01):
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            with self._lock:
                if self.current_trial is None:
                    return False
                if self.current_trial["epoch_complete"]:
                    return True
            time.sleep(max(0.001, float(poll_s)))
        return False

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        with self._lock:
            if self.dataset_x:
                self._save_dataset_locked()
        if self.dataset_saved:
            print(f"[metabci_stim] dataset saved: {self.dataset_path}, epochs={len(self.dataset_x)}")
        else:
            print("[metabci_stim] no complete 1-2 epochs captured; dataset not written.")
        for obj in [
            getattr(self, "trials_csv_file", None),
            getattr(self, "meta_csv_file", None),
            getattr(self, "trigger_send_sock", None),
            getattr(self, "eeg_tcp_sock", None),
        ]:
            if obj is None:
                continue
            try:
                obj.close()
            except Exception:
                pass


def run_ssvep_with_node4_recording(
    VSObject,
    win,
    bg_color,
    display_time=1.0,
    index_time=1.0,
    rest_time=0.5,
    response_time=2,
    port_addr=9045,
    nrep=1,
    pdim="ssvep",
    lsl_source_id=None,
    online=None,
    device_type="NeuroScan",
):
    """Custom SSVEP loop that keeps original flashing logic and records EEG+labels."""
    if pdim != "ssvep":
        raise ValueError("run_ssvep_with_node4_recording only supports pdim='ssvep'")

    if device_type == "NeuroScan":
        port = NeuroScanPort(port_addr, use_serial=True) if port_addr else None
    elif device_type == "Neuracle":
        port = NeuraclePort(port_addr) if port_addr else None
    else:
        raise KeyError(f"Unknown device type: {device_type}")

    win.color = bg_color
    fps = VSObject.refresh_rate
    port_frame = int(0.05 * fps)
    inlet = None

    if online and lsl_source_id:
        try:
            from pylsl import StreamInlet, resolve_byprop
        except Exception:
            StreamInlet = None
            resolve_byprop = None
        if StreamInlet is not None and resolve_byprop is not None:
            streams = resolve_byprop("source_id", lsl_source_id, timeout=5)
            if streams:
                inlet = StreamInlet(streams[0])

    recorder = Node4StyleEEGRecorder(
        ring_buffer_seconds=max(20.0, float(VSObject.stim_time) * 8.0)
    )
    recorder.start()
    trial_id = 0

    try:
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")

        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()

        if port:
            port.setData(0)

        for trial in trials:
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            trial_id += 1
            target_idx = int(trial["id"])
            target_id = target_idx + 1
            target_freq = float(VSObject.freqs[target_idx])
            recorder.start_trial(trial_id=trial_id, target_id=target_id, target_freq_hz=target_freq)

            position = VSObject.stim_pos[target_idx] + np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    iframe += 1
                    win.flip()

            recorder.on_stim_start()
            for sf in range(VSObject.stim_frames):
                if sf == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, target_idx + 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, target_idx + 1)
                if sf == port_frame and port:
                    port.setData(0)
                VSObject.flash_stimuli[sf].draw()
                win.flip()
            recorder.on_stim_end()
            recorder.wait_current_trial_epoch(timeout_s=0.8, poll_s=0.01)

            if inlet is not None:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()
                samples, _ = inlet.pull_sample()
                predict_id = int(samples[0]) - 1
                VSObject.symbol_text = VSObject.symbol_text + VSObject.symbols[predict_id]
                res_text_pos = (
                    VSObject.res_text_pos[0] + VSObject.symbol_height / 3,
                    VSObject.res_text_pos[1],
                )
                iframe = 0
                while iframe < int(fps * response_time):
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = VSObject.symbol_text
                    VSObject.text_response.pos = res_text_pos
                    VSObject.text_response.draw()
                    iframe += 1
                    win.flip()

            recorder.finalize_trial()
    finally:
        recorder.close()


if __name__ == "__main__":
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    mon.setSizePix([2560, 1440])  # 显示器的分辨率
    mon.save()
    bg_color_warm = np.array([0, 0, 0])
    win_size = np.array([2560, 1440])
    # esc/q退出开始选择界面
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0,
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=False,  # True全窗口,此时win_size参数默认屏幕分辨率
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()

    # q退出范式界面
    """
    SSVEP
    """
    n_elements, rows, columns = 20, 4, 5  # n_elements 指令数量;  rows 行;  columns 列
    stim_length, stim_width = 200, 200  # ssvep单指令的尺寸
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]  # 指令的颜色，文字的颜色
    fps = 240  # 屏幕刷新率
    stim_time = 2  # 刺激时长
    stim_opacities = 1  # 刺激对比度
    freqs = np.arange(8, 16, 0.4)  # 指令的频率
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位

    basic_ssvep = SSVEP(win=win)

    basic_ssvep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    basic_ssvep.config_text(tex_color=tex_color)
    basic_ssvep.config_color(
        refresh_rate=fps,
        stim_time=stim_time,
        stimtype="sinusoid",
        stim_color=stim_color,
        stim_opacities=stim_opacities,
        freqs=freqs,
        phases=phases,
    )
    basic_ssvep.config_index()
    basic_ssvep.config_response()

    bg_color = np.array([0.3, 0.3, 0.3])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 1  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    port_addr = "COM8"  #  0xdefc                                  # 采集主机端口
    port_addr = None  #  0xdefc
    nrep = 2  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "basic SSVEP",
        run_ssvep_with_node4_recording,
        VSObject=basic_ssvep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="ssvep",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    """
    AVEP
    """
    n_elements, rows, columns = 20, 5, 4  # n_elements 指令数量;  rows 行;  columns 列
    stim_length, stim_width = 3, 3  # avep刺激点的尺寸
    tex_height = 25  # avep指令的大小
    stim_color, tex_color = [0.7, 0.7, 0.7], [1, 1, 1]  # 指令的颜色，文字的颜色
    fps = 60  # 屏幕刷新率
    stim_time = 1  # 刺激时长
    stim_opacities = 1  # 刺激对比度
    freqs = 4  # 指令的频率
    # phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位
    stim_num = 2
    avep = AVEP(win=win, dot_shape="cluster")
    sequence = [avep.num2bin_ary(i, n_elements) for i in range(n_elements)]
    # sequence = [[1,2,3,4] for i in range(n_elements)]
    if len(sequence) != n_elements:
        raise Exception("Incorrect spatial code amount!")
    avep.tex_height = tex_height
    avep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    avep.config_color(
        refresh_rate=fps,
        stim_time=stim_time,
        stimtype="sinusoid",
        stim_color=stim_color,
        sequence=sequence,
        stim_opacities=stim_opacities,
        freqs=np.ones((n_elements)) * freqs,
        stim_num=stim_num,
    )

    avep.config_text(symbol_height=tex_height, tex_color=tex_color)
    avep.config_index(index_height=40)
    avep.config_response()

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 0.5  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    port_addr = None  # 0xdefc                                  # 采集主机端口
    nrep = 1  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "avep",
        paradigm,
        VSObject=avep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="avep",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    """
    P300
    """
    n_elements, rows, columns = 36, 6, 6  # n_elements 指令数量;  rows 行;  columns 列
    tex_color = [1, 1, 1]  # 文字的颜色
    fps = 240  # 屏幕刷新率
    stim_duration = 0.1
    stim_ISI = 0.075
    stim_round = 6  # 单指令刺激轮次
    basic_P300 = P300(win=win)
    basic_P300.config_pos(n_elements=n_elements, rows=rows, columns=columns)
    basic_P300.config_text(tex_color=tex_color)
    basic_P300.config_color(
        refresh_rate=fps,
        stim_duration=stim_duration,
        stim_ISI=stim_ISI,
        stim_round=stim_round,
    )
    basic_P300.config_index()
    basic_P300.config_response(bg_color=[0, 0, 0])

    bg_color = np.array([0, 0, 0])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 0.5  # 提示时长，转移视线
    response_time = 2  # 在线反馈
    rest_time = 0.5  # 提示后的休息时长
    port_addr = "COM8"  #  0xdefc                                  # 采集主机端口
    nrep = 1  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "basic P300",
        paradigm,
        VSObject=basic_P300,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="p300",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    """
    MI
    """
    fps = 240  # 屏幕刷新率
    text_pos = (0.0, 0.0)  # 提示文本位置
    left_pos = [[-480, 0.0]]  # 左手位置
    right_pos = [[480, 0.0]]  # 右手位置
    tex_color = 2 * np.array([179, 45, 0]) / 255 - 1  # 提示文本颜色
    normal_color = [[-0.8, -0.8, -0.8]]  # 默认颜色
    image_color = [[1, 1, 1]]  # 提示或开始想象颜色
    symbol_height = 100  # 提示文本的高度
    n_Elements = 1  # 左右手各一个
    stim_length = 288  # 长度
    stim_width = 288  # 宽度
    basic_MI = MI(win=win)
    basic_MI.config_color(
        refresh_rate=fps,
        text_pos=text_pos,
        left_pos=left_pos,
        right_pos=right_pos,
        tex_color=tex_color,
        normal_color=normal_color,
        image_color=image_color,
        symbol_height=symbol_height,
        n_Elements=n_Elements,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    basic_MI.config_response()

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 2  # 提示时长，转移视线
    rest_time = 1  # 提示后的休息时长
    image_time = 4  # 想象时长
    response_time = 2  # 在线反馈
    port_addr = "COM8"  #  0xdefc                                  # 采集主机端口
    nrep = 15  # block数目
    lsl_source_id = "meta_online_worker"  # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "basic MI",
        paradigm,
        VSObject=basic_MI,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        image_time=image_time,
        pdim="mi",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    """
    连续反馈，不设定反馈显示时长，线程获取预测标签 con-SSVEP
    """
    n_elements, rows, columns = 20, 4, 5  # n_elements 指令数量;  rows 行;  columns 列
    stim_length, stim_width = 150, 150  # ssvep单指令的尺寸
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]  # 指令的颜色，文字的颜色
    fps = 120  # 屏幕刷新率
    stim_time = 2  # 刺激时长
    stim_opacities = 1  # 刺激对比度
    freqs = np.arange(8, 16, 0.4)  # 指令的频率
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位

    basic_ssvep = SSVEP(win=win)

    basic_ssvep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    basic_ssvep.config_text(tex_color=tex_color)
    basic_ssvep.config_color(
        refresh_rate=fps,
        stim_time=stim_time,
        stimtype="sinusoid",
        stim_color=stim_color,
        stim_opacities=stim_opacities,
        freqs=freqs,
        phases=phases,
    )
    basic_ssvep.config_index()
    basic_ssvep.config_response()

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 0.5  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    port_addr = None  #  0xdefc                                  # 采集主机端口
    nrep = 1  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "continous SSVEP",
        paradigm,
        VSObject=basic_ssvep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="con-ssvep",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    """
    SSaVEP
    """
    n_elements, rows, columns = 20, 4, 5
    n_members = 8
    stim_length, stim_width = 150, 150
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]
    fps = 240
    stim_time_member = 0.5
    stim_opacities = [1]
    freqs = np.array(
        [4, 8, 12, 16, 20, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20]
    )
    phases = np.zeros((n_elements, 1))
    basic_code = [[0, 1], [2, 3], [4, 5], [6, 7]]
    code_sequences = [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
        [3, 0, 1, 2],
        [3, 0, 1, 2],
        [3, 0, 1, 2],
        [3, 0, 1, 2],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
    ]

    code = code_sequence_generate(basic_code, code_sequences)
    n_sequence = np.shape(code)[1]
    angles = np.zeros(n_elements)
    outter_deg = 4
    inner_deg = 1.5
    radius = deg2pix(outter_deg, mon) / win_size[1] * 0.7
    basic_ssavep = SSAVEP(win=win, n_elements=n_elements, n_members=n_members)
    basic_ssavep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    basic_ssavep.stim_width = pix2height(win_size, basic_ssavep.stim_width)
    basic_ssavep.config_member_pos(
        win,
        radius=radius,
        angles=angles,
        outter_deg=outter_deg,
        inner_deg=inner_deg,
        tex_pix=256,
        sep_line_pix=16,
    )
    basic_ssavep.config_text(tex_color=tex_color, unit="height", symbol_height=0.03)
    basic_ssavep.config_stim(
        win,
        sizes=[[basic_ssavep.radius * 0.9, basic_ssavep.radius * 0.9]],
        member_degree=None,
        stim_color=stim_color,
        stim_opacities=stim_opacities,
    )
    # win.close()

    basic_ssavep.config_flash_array(
        refresh_rate=fps,
        freqs=freqs,
        phases=phases,
        codes=code,
        stim_time_member=stim_time_member,
        stimtype="sinusoid",
        stim_color=stim_color,
    )
    basic_ssavep.config_color(
        win,
        refresh_rate=fps,
        freqs=freqs,
        phases=phases,
        codes=code,
        stim_time_member=stim_time_member,
        stimtype="sinusoid",
        stim_color=stim_color,
        sizes=[[basic_ssavep.radius * 0.9, basic_ssavep.radius * 0.9]],
    )
    basic_ssavep.config_ring(
        win,
        sizes=[[basic_ssavep.radius * 2.15, basic_ssavep.radius * 2.15]],
        ring_colors=[2 * np.array([160, 160, 160]) / 255 - 1],
        opacities=stim_opacities,
    )
    basic_ssavep.config_target(
        win,
        sizes=[[basic_ssavep.radius * 0.2, basic_ssavep.radius * 0.2]],
        target_colors=[1, 1, 0],
        opacities=stim_opacities,
    )
    basic_ssavep.config_index(index_height=0.08, units="height")
    basic_ssavep.config_response()

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = 0.5  # 范式开始1s的warm时长
    index_time = 1  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    # port_addr = 'COM8'  #  0xdefc                                  # 采集主机端口
    port_addr = None
    nrep = 2  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "basic SSaVEP",
        paradigm,
        VSObject=basic_ssavep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="ssavep",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    ex.run()
