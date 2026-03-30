"""
阿里云FunASR连接器

支持阿里云FunASR（达摩院语音识别）服务。
文档：https://help.aliyun.com/zh/model-studio/fun-asr-recorded-speech-recognition-restful-api
"""

import logging
import os
import httpx
import json
import time
from typing import Dict, Any, Set, Optional
from urllib.parse import urlparse, urlunparse

from ..base import (
    BaseTranscriptionConnector,
    TranscriptionCapability,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionSegment,
    ConnectorSpecifications,
)
from ..exceptions import TranscriptionError, ConfigurationError, ProviderError

logger = logging.getLogger(__name__)


class AlibabaFunASRConnector(BaseTranscriptionConnector):
    """阿里云FunASR连接器"""

    CAPABILITIES: Set[TranscriptionCapability] = {
        TranscriptionCapability.DIARIZATION,
        TranscriptionCapability.TIMESTAMPS,
        TranscriptionCapability.LANGUAGE_DETECTION,
        TranscriptionCapability.SPEAKER_COUNT_CONTROL,
    }
    PROVIDER_NAME = "alibaba_funasr"

    SPECIFICATIONS = ConnectorSpecifications(
        max_file_size_bytes=None,
        max_duration_seconds=None,
        handles_chunking_internally=True,  # 阿里云FunASR处理分块
    )

    def __init__(self, config: Dict[str, Any]):
        """
        初始化阿里云FunASR连接器

        Args:
            config: 配置字典，包含：
                - base_url: 阿里云FunASR基础URL (required)
                - api_key: API密钥 (required)
                - model: 模型名称 (default: "fun-asr")
                - timeout: 请求超时时间 (default: 1800)
                - diarize: 是否启用说话人分离 (default: True)
                - disfluency_removal_enabled: 过滤语气词 (default: True)
                - timestamp_alignment_enabled: 时间戳校准功能 (default: False)
                - language_hints: 语言提示数组 (default: [])
        """
        super().__init__(config)

        self.base_url = config['base_url'].rstrip('/')
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'fun-asr')
        self._config_timeout = config.get('timeout', 1800)
        self.default_diarize = config.get('diarize', True)
        self.disfluency_removal_enabled = config.get('disfluency_removal_enabled', True)
        self.timestamp_alignment_enabled = config.get('timestamp_alignment_enabled', False)
        self.language_hints = config.get('language_hints', [])

        # 验证配置
        self._validate_config()

    def _validate_config(self) -> None:
        """验证配置"""
        if not self.config.get('base_url'):
            raise ConfigurationError("base_url is required for Alibaba FunASR connector")
        if not self.config.get('api_key'):
            raise ConfigurationError("api_key is required for Alibaba FunASR connector")

    @property
    def timeout(self):
        """获取超时时间"""
        # 环境变量优先
        env_timeout = os.environ.get('ASR_TIMEOUT') or os.environ.get('asr_timeout_seconds')
        if env_timeout:
            try:
                return int(env_timeout)
            except (ValueError, TypeError):
                pass

        # 返回配置值
        return self._config_timeout

    def _clean_presigned_url(self, url: str) -> str:
        """
        清理预签名URL，移除查询参数
        
        阿里云FunASR可能不接受包含查询参数的URL，所以我们需要移除签名参数
        只保留基本的URL部分
        """
        try:
            parsed = urlparse(url)
            # 移除查询参数，只保留协议、域名和路径
            cleaned = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                '',  # 移除参数
                '',  # 移除查询
                ''   # 移除片段
            ))
            logger.info(f"Cleaned presigned URL: {url[:50]}... -> {cleaned}")
            return cleaned
        except Exception as e:
            logger.warning(f"Failed to clean presigned URL {url[:50]}...: {e}")
            return url

    def _poll_task_result(self, task_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        轮询异步任务结果
        
        Args:
            task_id: 任务ID
            headers: 请求头
            
        Returns:
            任务结果数据
        """
        # 任务查询端点 - 根据官方示例使用POST方法
        task_url = f"{self.base_url}/tasks/{task_id}"
        
        # 轮询配置 - 根据官方示例调整
        max_attempts = 300  # 最大尝试次数（300 * 0.1 = 30秒）
        poll_interval = 0.1  # 轮询间隔（秒），官方示例使用0.1秒
        
        logger.info(f"开始轮询任务结果，任务ID: {task_id}")
        
        with httpx.Client() as client:
            for attempt in range(max_attempts):
                try:
                    if attempt % 10 == 0:  # 每10次记录一次日志
                        logger.info(f"轮询尝试 {attempt + 1}/{max_attempts}")
                    
                    # 根据官方示例，任务查询使用POST方法
                    response = client.post(task_url, headers=headers, timeout=30.0)
                    response.raise_for_status()
                    
                    data = response.json()
                    task_status = data.get('output', {}).get('task_status', 'UNKNOWN')
                    
                    if attempt % 10 == 0:  # 每10次记录一次状态
                        logger.info(f"任务状态: {task_status}")
                    
                    # 检查任务状态
                    if task_status == 'SUCCEEDED':
                        logger.info(f"任务完成，返回结果")
                        return data
                    elif task_status == 'FAILED':
                        # 获取详细的错误信息
                        output = data.get('output', {})
                        error_code = output.get('code', 'UNKNOWN_ERROR')
                        error_msg = output.get('message', '任务失败')
                        
                        # 记录完整的响应以便调试
                        logger.debug(f"阿里云完整响应: {json.dumps(data, indent=2)}")
                        logger.error(f"阿里云错误代码: {error_code}, 消息: {error_msg}")
                        
                        # 特殊处理ASR_RESPONSE_HAVE_NO_WORDS错误
                        if error_code == 'ASR_RESPONSE_HAVE_NO_WORDS':
                            raise TranscriptionError(
                                "音频文件中未检测到语音。请检查：\n"
                                "1. 音频文件是否包含清晰的语音\n"
                                "2. 音频音量是否足够\n"
                                "3. 音频格式是否正确（推荐WAV格式，16kHz采样率，单声道）\n"
                                "4. 尝试使用阿里云官方示例音频测试"
                            )
                        
                        raise ProviderError(
                            f"阿里云FunASR任务失败: {error_msg} (错误代码: {error_code})",
                            provider=self.PROVIDER_NAME,
                            status_code=response.status_code
                        )
                    elif task_status == 'PENDING' or task_status == 'RUNNING':
                        # 任务还在处理中，继续等待
                        time.sleep(poll_interval)
                    else:
                        # 未知状态
                        logger.warning(f"未知任务状态: {task_status}")
                        time.sleep(poll_interval)
                        
                except httpx.HTTPStatusError as e:
                    logger.error(f"任务查询失败，状态码: {e.response.status_code}")
                    if attempt < max_attempts - 1:
                        time.sleep(poll_interval)
                    else:
                        raise
                except Exception as e:
                    logger.error(f"任务查询异常: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(poll_interval)
                    else:
                        raise
        
        # 如果达到最大尝试次数仍未完成
        raise TranscriptionError(f"任务轮询超时，任务ID: {task_id}")

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """
        使用阿里云FunASR进行语音识别

        Args:
            request: 标准化转录请求

        Returns:
            TranscriptionResponse 包含分段和说话人信息
        """
        try:
            # 阿里云FunASR端点 - 根据官方文档使用正确的URL
            url = f"{self.base_url}/services/audio/asr/transcription"

            # 准备请求头 - 根据官方文档添加X-DashScope-Async头
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'X-DashScope-Async': 'enable'  # 必须添加此头以启用异步处理
            }

            # 准备请求体 - 根据官方API文档格式
            # 基础payload结构
            payload = {
                'model': self.model
            }
            
            # 处理文件URL或本地文件
            if request.file_urls and len(request.file_urls) > 0:
                # 使用文件URL - 根据官方文档格式
                file_url = request.file_urls[0]
                
                # 检查URL是否过期（如果是预签名URL）
                # 注意：这里我们无法重新上传文件，因为需要访问录音对象
                # 这个检查应该在调用transcribe之前完成
                # 这里只记录警告
                if 'Expires=' in file_url:
                    import re
                    try:
                        match = re.search(r'Expires=(\d+)', file_url)
                        if match:
                            expires_timestamp = int(match.group(1))
                            import time
                            current_time = int(time.time())
                            if expires_timestamp < current_time:
                                logger.warning(f"文件URL已过期: {file_url[:100]}...")
                                logger.warning("过期URL可能导致阿里云FunASR下载失败")
                            elif expires_timestamp < current_time + 86400:  # 24小时内过期（七牛云预签名URL有效期为24小时）
                                logger.warning(f"文件URL将在24小时内过期: {file_url[:100]}...")
                    except Exception as e:
                        logger.warning(f"无法检查URL过期时间: {e}")
                
                # 根据错误日志，阿里云可能无法下载文件
                # 尝试两种方式：原始URL和清理后的URL
                # 首先尝试原始URL（包含查询参数）
                payload['input'] = {
                    'file_urls': [file_url]
                }
                
                logger.info(f"使用原始文件URL进行阿里云FunASR转录: {file_url[:100]}...")
                logger.warning(f"注意：如果FILE_DOWNLOAD_FAILED，可能需要检查URL可访问性或使用公开可访问的URL")
            else:
                # 使用本地文件上传 - 阿里云FunASR可能不支持直接文件上传
                # 需要先将文件上传到对象存储，然后使用URL
                logger.error(f"阿里云FunASR需要文件URL，不支持本地文件上传")
                raise TranscriptionError(
                    "阿里云FunASR需要文件URL。请先将文件上传到对象存储。"
                )
            
            # 准备parameters字段 - 根据官方文档
            parameters = {}
            
            # 添加语言设置
            if request.language:
                parameters['language'] = request.language
                logger.info(f"使用转录语言: {request.language}")
            
            # 说话人分离设置
            should_diarize = request.diarize if request.diarize is not None else self.default_diarize
            if should_diarize:
                parameters['diarization_enabled'] = True
            
            # 说话人数量提示
            if request.min_speakers or request.max_speakers:
                # 阿里云FunASR使用speaker_count参数
                if request.max_speakers:
                    parameters['speaker_count'] = request.max_speakers
                elif request.min_speakers:
                    parameters['speaker_count'] = request.min_speakers
            
            # 提示词和热词
            if request.prompt:
                parameters['special_word_filter'] = request.prompt
            if request.hotwords:
                # 阿里云FunASR可能不支持热词，或者需要特殊格式
                logger.warning(f"阿里云FunASR可能不支持热词: {request.hotwords}")
            
            # 添加新的接口参数
            # 过滤语气词
            if self.disfluency_removal_enabled:
                parameters['disfluency_removal_enabled'] = True
            
            # 时间戳校准功能
            if self.timestamp_alignment_enabled:
                parameters['timestamp_alignment_enabled'] = True
            
            # 语言提示数组
            if self.language_hints:
                parameters['language_hints'] = self.language_hints
            
            # 如果有参数，添加到payload
            if parameters:
                payload['parameters'] = parameters
            
            # 发送JSON请求
            timeout = httpx.Timeout(
                None,
                connect=60.0,
                read=float(self.timeout),
                write=float(self.timeout),
                pool=None
            )
            
            logger.info(f"发送阿里云FunASR请求到 {url}，payload: {json.dumps(payload, indent=2)[:200]}...")
            
            with httpx.Client() as client:
                # 提交异步任务
                response = client.post(url, json=payload, headers=headers, timeout=timeout)
                logger.info(f"阿里云FunASR请求完成，状态码: {response.status_code}")
                response.raise_for_status()
                
                # 解析响应
                response_text = response.text
                try:
                    task_data = response.json()
                except Exception as json_err:
                    if response_text.strip().startswith('<'):
                        logger.error(f"阿里云FunASR返回HTML错误页面 (状态码 {response.status_code})")
                        raise ProviderError(
                            f"阿里云FunASR服务返回HTML错误页面",
                            provider=self.PROVIDER_NAME,
                            status_code=response.status_code
                        )
                    else:
                        raise ProviderError(
                            f"阿里云FunASR服务返回无效响应: {json_err}",
                            provider=self.PROVIDER_NAME,
                            status_code=response.status_code
                        )
                
                # 检查是否返回了任务ID
                task_id = task_data.get('output', {}).get('task_id')
                if not task_id:
                    logger.error(f"未找到任务ID，响应: {task_data}")
                    raise ProviderError(
                        f"阿里云FunASR未返回任务ID",
                        provider=self.PROVIDER_NAME,
                        status_code=response.status_code
                    )
                
                logger.info(f"收到异步任务ID: {task_id}")
                
                # 轮询任务结果
                result_data = self._poll_task_result(task_id, headers)
                
                # 解析最终结果
                return self._parse_response(result_data)

        except httpx.HTTPStatusError as e:
            logger.error(f"阿里云FunASR请求失败，状态码 {e.response.status_code}")
            error_detail = ""
            try:
                error_response = e.response.json()
                error_detail = f": {json.dumps(error_response)}"
            except:
                error_detail = f": {e.response.text[:200]}"
            
            raise ProviderError(
                f"阿里云FunASR请求失败，状态码 {e.response.status_code}{error_detail}",
                provider=self.PROVIDER_NAME,
                status_code=e.response.status_code
            ) from e

        except httpx.TimeoutException as e:
            logger.error(f"阿里云FunASR请求超时，超时时间 {self.timeout}s")
            raise TranscriptionError(f"阿里云FunASR请求超时，超时时间 {self.timeout}s") from e

        except Exception as e:
            error_msg = str(e)
            logger.error(f"阿里云FunASR转录失败: {error_msg}")
            raise TranscriptionError(f"阿里云FunASR转录失败: {error_msg}") from e

    def _download_transcription_result(self, transcription_url: str) -> Dict[str, Any]:
        """
        下载转录结果
        
        Args:
            transcription_url: 转录结果URL
            
        Returns:
            转录结果数据
        """
        try:
            logger.info(f"下载转录结果: {transcription_url[:100]}...")
            
            with httpx.Client(timeout=30.0) as client:
                response = client.get(transcription_url)
                response.raise_for_status()
                
                result_data = response.json()
                logger.info(f"转录结果下载成功，包含键: {list(result_data.keys())}")
                
                return result_data
                
        except Exception as e:
            logger.error(f"下载转录结果失败: {e}")
            raise TranscriptionError(f"下载转录结果失败: {e}")

    def _parse_response(self, data: Dict[str, Any]) -> TranscriptionResponse:
        """
        解析阿里云FunASR响应为标准化格式

        Args:
            data: 阿里云FunASR响应数据

        Returns:
            标准化的转录响应
        """
        logger.info(f"阿里云FunASR响应键: {list(data.keys())}")
        
        # 阿里云FunASR的响应结构
        output = data.get('output', {})
        
        # 检查是否有错误
        task_status = output.get('task_status')
        if task_status == 'FAILED':
            error_msg = output.get('message', '任务失败')
            raise ProviderError(
                f"阿里云FunASR任务失败: {error_msg}",
                provider=self.PROVIDER_NAME
            )
        
        # 根据官方示例，结果可能在results字段中
        results = output.get('results', [])
        
        # 如果没有results字段，尝试result字段（旧格式）
        if not results and 'result' in output:
            # 旧格式：单个结果
            result = output.get('result', {})
            results = [result]
        
        segments = []
        speakers = set()
        full_text_parts = []
        
        # 处理所有结果
        for result_idx, result in enumerate(results):
            # 检查子任务状态
            subtask_status = result.get('subtask_status')
            if subtask_status == 'FAILED':
                error_code = result.get('code', 'UNKNOWN_ERROR')
                error_msg = result.get('message', '子任务失败')
                logger.warning(f"子任务 {result_idx} 失败: {error_code} - {error_msg}")
                continue
            
            # 获取转录文本
            text = result.get('text', '')
            
            # 如果有transcription_url，下载转录结果
            transcription_url = result.get('transcription_url')
            if transcription_url:
                logger.info(f"子任务 {result_idx} 有转录URL: {transcription_url[:100]}...")
                
                try:
                    # 下载转录结果
                    transcription_data = self._download_transcription_result(transcription_url)
                    
                    # 解析转录数据
                    transcripts = transcription_data.get('transcripts', [])
                    
                    for transcript in transcripts:
                        # 获取文本
                        transcript_text = transcript.get('text', '')
                        if transcript_text:
                            text = transcript_text
                        
                        # 解析句子
                        transcript_sentences = transcript.get('sentences', [])
                        
                        for sentence in transcript_sentences:
                            sentence_text = sentence.get('text', '').strip()
                            if not sentence_text:
                                continue
                            
                            # 获取说话人ID
                            speaker_id = sentence.get('speaker_id', 0)
                            speaker = f'SPEAKER_{speaker_id:02d}'
                            
                            # 时间戳（毫秒转换为秒）
                            begin_time = sentence.get('begin_time', 0)
                            end_time = sentence.get('end_time', 0)
                            start_time = begin_time / 1000.0 if begin_time else 0
                            end_time = end_time / 1000.0 if end_time else 0
                            
                            speakers.add(speaker)
                            full_text_parts.append(f"[{speaker}]: {sentence_text}")
                            
                            segments.append(TranscriptionSegment(
                                text=sentence_text,
                                speaker=speaker,
                                start_time=start_time,
                                end_time=end_time
                            ))
                    
                except Exception as e:
                    logger.error(f"下载或解析转录结果失败: {e}")
                    # 继续处理其他结果
            
            # 如果没有从转录URL获取到数据，尝试直接解析结果
            if not segments and text:
                # 只有完整文本，没有分段
                segments.append(TranscriptionSegment(
                    text=text,
                    speaker='SPEAKER_00',
                    start_time=0,
                    end_time=None
                ))
                speakers.add('SPEAKER_00')
                full_text_parts.append(f"[SPEAKER_00]: {text}")
            
            # 解析句子（如果直接返回在结果中）
            sentences = result.get('sentences', [])
            if sentences and not segments:  # 只有在没有从转录URL获取数据时才使用
                # 有句子级别的分段
                for i, sentence in enumerate(sentences):
                    # 阿里云FunASR的说话人标签格式
                    speaker = sentence.get('spk', f'SPEAKER_{i:02d}')
                    
                    # 处理文本
                    sentence_text = sentence.get('text', '').strip()
                    if not sentence_text:
                        continue
                        
                    # 时间戳
                    start_time = sentence.get('start_time', 0)
                    end_time = sentence.get('end_time', 0)
                    
                    speakers.add(speaker)
                    full_text_parts.append(f"[{speaker}]: {sentence_text}")
                    
                    segments.append(TranscriptionSegment(
                        text=sentence_text,
                        speaker=speaker,
                        start_time=start_time,
                        end_time=end_time
                    ))
        
        # 如果没有解析到任何内容
        if not segments:
            # 尝试从output中获取文本
            if 'result' in output:
                result = output['result']
                text = result.get('text', '')
                if text:
                    segments.append(TranscriptionSegment(
                        text=text,
                        speaker='SPEAKER_00',
                        start_time=0,
                        end_time=None
                    ))
                    speakers.add('SPEAKER_00')
                    full_text_parts.append(f"[SPEAKER_00]: {text}")
                else:
                    # 未知格式
                    logger.warning(f"未知的阿里云FunASR响应格式: {data}")
                    full_text = json.dumps(data)
                    segments.append(TranscriptionSegment(
                        text=full_text,
                        speaker='SPEAKER_00',
                        start_time=0,
                        end_time=None
                    ))
                    speakers.add('SPEAKER_00')
                    full_text_parts.append(f"[SPEAKER_00]: {full_text}")
        
        # 获取完整文本
        if full_text_parts:
            full_text = '\n'.join(full_text_parts)
        else:
            full_text = ''
        
        # 获取语言 - 尝试从第一个结果中获取
        language = 'zh-CN'  # 默认值
        if results and len(results) > 0:
            language = results[0].get('language', 'zh-CN')
        elif 'result' in output:
            language = output['result'].get('language', 'zh-CN')
        
        logger.info(f"解析了 {len(segments)} 个分段，包含 {len(speakers)} 个不同的说话人: {sorted(speakers)}")
        
        return TranscriptionResponse(
            text=full_text,
            segments=segments,
            speakers=sorted(list(speakers)),
            speaker_embeddings=None,  # 阿里云FunASR不支持说话人嵌入
            language=language,
            provider=self.PROVIDER_NAME,
            model=self.model,
            raw_response=data
        )

    def health_check(self) -> bool:
        """检查阿里云FunASR服务是否可达"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            with httpx.Client(timeout=10.0) as client:
                # 尝试健康检查端点
                health_url = f"{self.base_url}/health"
                try:
                    response = client.get(health_url, headers=headers)
                    if response.status_code < 500:
                        return True
                except Exception:
                    pass
                
                # 尝试基础端点
                try:
                    response = client.get(self.base_url, headers=headers)
                    if response.status_code < 500:
                        return True
                except Exception:
                    pass
                
                # 尝试服务端点
                try:
                    service_url = f"{self.base_url}/services/audio/asr/transcription"
                    response = client.get(service_url, headers=headers)
                    if response.status_code < 500:
                        return True
                except Exception:
                    pass
                
                return False
        except Exception:
            return False

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """返回配置的JSON模式"""
        return {
            "type": "object",
            "required": ["base_url", "api_key"],
            "properties": {
                "base_url": {
                    "type": "string",
                    "description": "阿里云FunASR基础URL (e.g., https://dashscope.aliyuncs.com/api/v1)"
                },
                "api_key": {
                    "type": "string",
                    "description": "阿里云API密钥"
                },
                "model": {
                    "type": "string",
                    "default": "fun-asr",
                    "description": "FunASR模型名称"
                },
                "timeout": {
                    "type": "integer",
                    "default": 1800,
                    "description": "请求超时时间（秒）"
                },
                "diarize": {
                    "type": "boolean",
                    "default": True,
                    "description": "启用说话人分离"
                },
                "disfluency_removal_enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "过滤语气词"
                },
                "timestamp_alignment_enabled": {
                    "type": "boolean",
                    "default": False,
                    "description": "启用时间戳校准功能"
                },
                "language_hints": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "default": [],
                    "description": "语言提示数组"
                }
            }
        }
