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
        # 优先使用新的环境变量，如果未设置则使用配置中的值
        base_url = config.get('base_url', '')
        api_key = config.get('api_key', '')
        model = config.get('model', 'fun-asr')
        
        # 检查新的环境变量
        if not base_url:
            base_url = os.environ.get('FUNASR_BASE_URL', '')
            if base_url:
                logger.info(f"使用FUNASR_BASE_URL环境变量: {base_url}")
        
        if not api_key:
            api_key = os.environ.get('FUNASR_API_KEY', '')
            if api_key:
                logger.info(f"使用FUNASR_API_KEY环境变量")
        
        # 检查FUNASR_MODEL环境变量
        funasr_model = os.environ.get('FUNASR_MODEL', '')
        if funasr_model:
            model = funasr_model
            logger.info(f"使用FUNASR_MODEL环境变量: {model}")
        
        self.base_url = base_url.rstrip('/') if base_url else ''
        self.api_key = api_key
        self.model = model
        self._config_timeout = config.get('timeout', 1800)
        self.default_diarize = config.get('diarize', True)
        self.disfluency_removal_enabled = config.get('disfluency_removal_enabled', True)
        self.timestamp_alignment_enabled = config.get('timestamp_alignment_enabled', False)
        self.language_hints = config.get('language_hints', [])

        # 调用父类初始化，这会调用_validate_config
        super().__init__(config)

    def _validate_config(self) -> None:
        """验证配置"""
        # 最终验证
        if not self.base_url:
            raise ConfigurationError(
                "base_url is required for Alibaba FunASR connector. "
                "Set FUNASR_BASE_URL or ASR_BASE_URL environment variable, "
                "or provide base_url in config."
            )
        if not self.api_key:
            raise ConfigurationError(
                "api_key is required for Alibaba FunASR connector. "
                "Set FUNASR_API_KEY or TRANSCRIPTION_API_KEY environment variable, "
                "or provide api_key in config."
            )

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
                        
                        # 特殊处理SUCCESS_WITH_NO_VALID_FRAGMENT错误
                        # 这个错误代码表示任务成功但没有有效的音频片段
                        # 应该返回空结果而不是抛出错误
                        if error_code == 'SUCCESS_WITH_NO_VALID_FRAGMENT':
                            logger.warning(f"阿里云FunASR任务成功但没有有效的音频片段: {error_msg}")
                            # 返回一个空的成功响应
                            return {
                                'output': {
                                    'task_status': 'SUCCEEDED',
                                    'results': [],
                                    'message': error_msg
                                }
                            }
                        
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
                logger.info(f"成功下载转录结果，数据大小: {len(str(result_data))} 字符")
                return result_data
                
        except httpx.HTTPStatusError as e:
            logger.error(f"下载转录结果失败，状态码: {e.response.status_code}")
            raise ProviderError(
                f"下载转录结果失败，状态码: {e.response.status_code}",
                provider=self.PROVIDER_NAME,
                status_code=e.response.status_code
            ) from e
        except Exception as e:
            logger.error(f"下载转录结果异常: {e}")
            raise TranscriptionError(f"下载转录结果失败: {e}") from e

    def _parse_response(self, data: Dict[str, Any]) -> TranscriptionResponse:
        """
        解析阿里云FunASR响应
        
        Args:
            data: 阿里云FunASR返回的JSON数据
            
        Returns:
            标准化的TranscriptionResponse
        """
        try:
            output = data.get('output', {})
            task_status = output.get('task_status', 'UNKNOWN')
            
            # 检查任务状态
            if task_status != 'SUCCEEDED':
                error_code = output.get('code', 'UNKNOWN_ERROR')
                error_msg = output.get('message', '任务失败')
                
                # 特殊处理SUCCESS_WITH_NO_VALID_FRAGMENT错误
                if error_code == 'SUCCESS_WITH_NO_VALID_FRAGMENT':
                    logger.warning(f"阿里云FunASR任务成功但没有有效的音频片段: {error_msg}")
                    # 返回空的转录结果
                    return TranscriptionResponse(
                        text='',
                        segments=[],
                        speakers=[],
                        language=None,
                        provider=self.PROVIDER_NAME,
                        model=self.model,
                        raw_response=data
                    )
                
                # 其他错误应该已经在_poll_task_result中处理了
                raise ProviderError(
                    f"阿里云FunASR任务失败: {error_msg} (错误代码: {error_code})",
                    provider=self.PROVIDER_NAME,
                    status_code=200  # 阿里云返回200状态码但任务失败
                )
            
            # 解析成功的结果
            results = output.get('results', [])
            segments = []
            speakers = set()
            full_text_parts = []
            
            logger.info(f"解析阿里云FunASR响应，结果数量: {len(results)}")
            
            for result in results:
                # 阿里云FunASR返回的格式可能包含多个句子
                sentences = result.get('sentences', [])
                
                for sentence in sentences:
                    text = sentence.get('text', '').strip()
                    if not text:
                        continue
                    
                    # 获取说话人信息
                    speaker = sentence.get('speaker', 'SPEAKER_0')
                    
                    # 标准化说话人格式
                    if speaker.startswith('SPEAKER_'):
                        speaker_id = speaker
                    else:
                        # 尝试提取说话人ID
                        try:
                            speaker_id = f"SPEAKER_{int(speaker)}"
                        except (ValueError, TypeError):
                            speaker_id = f"SPEAKER_{speaker}"
                    
                    # 获取时间戳
                    start_time = sentence.get('start_time', 0.0)
                    end_time = sentence.get('end_time', 0.0)
                    
                    # 确保时间戳是浮点数
                    try:
                        start_time = float(start_time)
                    except (ValueError, TypeError):
                        start_time = 0.0
                    
                    try:
                        end_time = float(end_time)
                    except (ValueError, TypeError):
                        end_time = 0.0
                    
                    # 创建分段
                    segment = TranscriptionSegment(
                        text=text,
                        speaker=speaker_id,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    segments.append(segment)
                    speakers.add(speaker_id)
                    full_text_parts.append(f"[{speaker_id}]: {text}")
            
            # 构建完整文本
            if full_text_parts:
                full_text = '\n'.join(full_text_parts)
            else:
                full_text = ''
            
            # 获取语言信息
            language = None
            for result in results:
                if 'language' in result:
                    language = result.get('language')
                    break
            
            logger.info(f"解析完成: {len(segments)}个分段, {len(speakers)}个说话人")
            
            return TranscriptionResponse(
                text=full_text,
                segments=segments,
                speakers=sorted(list(speakers)),
                language=language,
                provider=self.PROVIDER_NAME,
                model=self.model,
                raw_response=data
            )
            
        except Exception as e:
            logger.error(f"解析阿里云FunASR响应失败: {e}")
            raise TranscriptionError(f"解析阿里云FunASR响应失败: {e}") from e

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """返回配置的JSON Schema"""
        return {
            "type": "object",
            "required": ["base_url", "api_key"],
            "properties": {
                "base_url": {
                    "type": "string",
                    "description": "阿里云FunASR基础URL (例如: https://dashscope.aliyuncs.com/api/v1)"
                },
                "api_key": {
                    "type": "string",
                    "description": "阿里云FunASR API密钥"
                },
                "model": {
                    "type": "string",
                    "default": "fun-asr",
                    "description": "模型名称"
                },
                "timeout": {
                    "type": "integer",
                    "default": 1800,
                    "description": "请求超时时间（秒）"
                },
                "diarize": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否启用说话人分离"
                },
                "disfluency_removal_enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否过滤语气词"
                },
                "timestamp_alignment_enabled": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否启用时间戳校准功能"
                },
                "language_hints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "语言提示数组"
                }
            }
        }
