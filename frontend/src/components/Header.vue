<template>
    <div class="title">
        <a href="https://github.com/hahnyuan/LLM-Viewer" target="_blank" class="hover-bold">LLM-Viewer</a>
        v{{ version }}
    </div>
    <div class="header_button">
        |
        <span>模型: </span>
        <select v-model="select_model_id">
            <option v-for="model_id in avaliable_model_ids" :value="model_id">{{ model_id }}</option>
        </select>
        <span> | </span>
        <span>硬件: </span>
        <select v-model="select_hardware">
            <option v-for="hardware in avaliable_hardwares" :value="hardware">{{ hardware }}</option>
        </select>
    </div>
    <div>
        <span> | </span>
        <span>服务器: </span>
        <select v-model="ip_port">
            <option value="api.llm-viewer.com">api.llm-viewer.com</option>
            <option value="127.0.0.1:5050">127.0.0.1</option>
        </select>
    </div>
    <div>
        <span> | </span>
        <span class="hover-bold" @click="is_show_help = ! is_show_help">帮助</span>
    </div>
    <div>
        <span> | </span>
        <a href="https://github.com/hahnyuan/LLM-Viewer" target="_blank" class="hover-bold">Github 项目</a>
    </div>
    <div>
        <span> | </span>
        <a href="https://arxiv.org/pdf/2402.16363.pdf" target="_blank" class="hover-bold">论文</a>
    </div>
    <div v-if="is_show_help" class="float-info-window">
        <!-- item -->
        <p>LLM-Viewer 是一个开源工具，用于可视化 LLM 模型并分析其在硬件设备上的部署性能。</p>
        <p>
            在页面中央，你可以看到 LLM 模型图。点击节点可查看该节点的详细信息。
        </p>
        <p>↑ 在页面顶部，你可以设置 LLM 模型、硬件设备和服务器。
            如果你在本地部署了 LLM-Viewer，可以选择 localhost 服务器。
        </p>
        <p>
            ← 在页面左侧，你可以看到配置面板。可设置推理配置和优化配置。
        </p>
        <p>
            ↙ 左侧面板会展示网络级分析结果。
        </p>
        <p>
            欢迎阅读我们的论文 <a class="hover-bold" href="https://arxiv.org/pdf/2402.16363.pdf" target="_blank">LLM Inference Unveiled: Survey and Roofline Model Insights</a>。
            在这篇论文中，我们基于 LLM-Viewer 对高效 LLM 推理的最新进展进行了系统分析。
            引用 BibTeX：
        </p>
        @article{yuan2024llm,<br/>
            &nbsp    title={LLM Inference Unveiled: Survey and Roofline Model Insights},<br/>
            &nbsp    author={Yuan, Zhihang and Shang, Yuzhang and Zhou, Yang and Dong, Zhen and Xue, Chenhao and Wu, Bingzhe and Li, Zhikai and Gu, Qingyi and Lee, Yong Jae and Yan, Yan and others},<br/>
            &nbsp    journal={arXiv preprint arXiv:2402.16363},<br/>
            &nbsp    year={2024}<br/>
        }
    </div>
</template>

<script setup>
import { inject, ref, watch, computed, onMounted } from 'vue';
import axios from 'axios'
const model_id = inject('model_id');
const hardware = inject('hardware');
const global_update_trigger = inject('global_update_trigger');
const ip_port = inject('ip_port');

const avaliable_hardwares = ref([]);
const avaliable_model_ids = ref([]);

const version = ref(llm_viewer_frontend_version)

const is_show_help = ref(false)

function update_avaliable() {
    const url = 'http://' + ip_port.value + '/get_avaliable'
    axios.get(url).then(function (response) {
        console.log(response);
        avaliable_hardwares.value = response.data.avaliable_hardwares
        avaliable_model_ids.value = response.data.avaliable_model_ids
    })
        .catch(function (error) {
            console.log("获取可用配置失败");
            console.log(error);
        });
}

onMounted(() => {
    console.log("Header 已挂载")
    update_avaliable()
})

var select_model_id = ref('meta-llama/Llama-2-7b-hf');
watch(select_model_id, (n) => {
    console.log("select_model_id", n)
    model_id.value = n
    global_update_trigger.value += 1
})

var select_hardware = ref('nvidia_V100');
watch(select_hardware, (n) => {
    console.log("select_hardware", n)
    hardware.value = n
    global_update_trigger.value += 1
})

watch(ip_port, (n) => {
    console.log("ip_port", n)
    update_avaliable()
})


</script>

<style scoped>
.header_button button {
    font-size: 1.0rem;
    margin: 5px;
    padding: 5px;
    border-radius: 5px;
    border: 1px solid #000000;
    /* background-color: #fff; */
    /* color: #000; */
    cursor: pointer;
}

.header_button button:hover {
    color: #fff;
    background-color: #000;
}

.header_button button:active {
    color: #fff;
    background-color: #000;
}

.active {
    color: #fff;
    background-color: #5b5b5b;
}



.title {
    font-size: 18px;
    /* 左对齐 */
    text-align: left;
}

.hover-bold{
    color: inherit;
    /* text-decoration: none; */
}

.hover-bold:hover {
    font-weight: bold;
}


.float-info-window {
    position: absolute;
    top: 80px;
    left: 40%;
    height: auto;
    width: 30%;
    background-color: #f1f1f1ed;
    padding: 20px;
    /* background-color: #fff; */
    /* border: 2px solid #4e4e4e; */
    z-index: 999;
}
</style>