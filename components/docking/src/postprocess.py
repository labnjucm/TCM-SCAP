"""
后处理模块 - 仅用于推理

处理推理结果：格式化输出、生成报告等
"""

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional


def format_result_summary(result: Dict[str, Any]) -> str:
    """
    格式化推理结果摘要
    
    Args:
        result: 推理结果字典
    
    Returns:
        格式化的摘要字符串
    """
    if not result.get('success', False):
        error_msg = result.get('error', '未知错误')
        return f"❌ 推理失败\n错误: {error_msg}"
    
    lines = []
    lines.append("✅ 推理成功!")
    lines.append(f"\n复合物: {result.get('complex_name', 'N/A')}")
    lines.append(f"输出目录: {result.get('output_dir', 'N/A')}")
    
    confidences = result.get('confidences', [])
    if confidences and len(confidences) > 0:
        lines.append(f"\n生成了 {len(confidences)} 个样本:")
        # 显示前5个样本的置信度
        for i, conf in enumerate(confidences[:5]):
            lines.append(f"  Rank {i+1}: 置信度 = {conf:.3f}")
        
        if len(confidences) > 5:
            lines.append(f"  ... 以及其他 {len(confidences)-5} 个样本")
    
    files = result.get('files', [])
    if files:
        lines.append(f"\n生成了 {len(files)} 个输出文件")
    
    return "\n".join(lines)


def create_result_zip(output_dir: str, zip_name: Optional[str] = None) -> str:
    """
    将结果打包成ZIP文件
    
    Args:
        output_dir: 输出目录
        zip_name: ZIP文件名（可选）
    
    Returns:
        ZIP文件路径
    """
    if zip_name is None:
        zip_name = f"{Path(output_dir).name}_results.zip"
    
    zip_path = os.path.join(Path(output_dir).parent, zip_name)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, Path(output_dir).parent)
                zipf.write(file_path, arcname)
    
    return zip_path


def generate_batch_report(results: List[Dict[str, Any]]) -> str:
    """
    生成批量推理报告
    
    Args:
        results: 推理结果列表
    
    Returns:
        报告字符串
    """
    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    failed = total - successful
    
    lines = []
    lines.append("=" * 50)
    lines.append("批量推理报告")
    lines.append("=" * 50)
    lines.append(f"总计: {total} 个复合物")
    lines.append(f"成功: {successful} 个 ({successful/total*100:.1f}%)")
    lines.append(f"失败: {failed} 个 ({failed/total*100:.1f}%)")
    lines.append("")
    
    if failed > 0:
        lines.append("失败的复合物:")
        for i, result in enumerate(results):
            if not result.get('success', False):
                name = result.get('complex_name', f'complex_{i}')
                error = result.get('error', 'N/A')
                lines.append(f"  - {name}: {error}")
        lines.append("")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


def extract_top_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    提取最佳结果（rank1）
    
    Args:
        result: 完整的推理结果
    
    Returns:
        包含最佳结果信息的字典
    """
    if not result.get('success', False):
        return result
    
    output_dir = result.get('output_dir', '')
    rank1_file = os.path.join(output_dir, 'rank1.sdf')
    
    confidence = None
    if result.get('confidences'):
        confidence = result['confidences'][0]
    
    return {
        'success': True,
        'complex_name': result.get('complex_name'),
        'rank1_file': rank1_file if os.path.exists(rank1_file) else None,
        'confidence': confidence,
        'output_dir': output_dir
    }

