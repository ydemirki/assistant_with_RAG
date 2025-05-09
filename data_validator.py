import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class ExcelDataValidator:
    """Excel veri doğrulama ve temizleme sınıfı"""
    
    def __init__(self):
        self.validation_rules = {
            'numeric': lambda x: pd.to_numeric(x, errors='coerce').notna(),
            'date': lambda x: pd.to_datetime(x, errors='coerce').notna(),
            'text': lambda x: x.astype(str).str.len() > 0,
            'email': lambda x: x.astype(str).str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'phone': lambda x: x.astype(str).str.match(r'^\+?[\d\s-]{10,}$')
        }
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Sütun tiplerini otomatik tespit et"""
        column_types = {}
        
        for column in df.columns:
            # Sayısal değer kontrolü
            if df[column].apply(lambda x: isinstance(x, (int, float)) or 
                              (isinstance(x, str) and x.replace('.', '').isdigit())).all():
                column_types[column] = 'numeric'
            
            # Tarih kontrolü
            elif pd.to_datetime(df[column], errors='coerce').notna().all():
                column_types[column] = 'date'
            
            # E-posta kontrolü
            elif df[column].astype(str).str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$').all():
                column_types[column] = 'email'
            
            # Telefon kontrolü
            elif df[column].astype(str).str.match(r'^\+?[\d\s-]{10,}$').all():
                column_types[column] = 'phone'
            
            # Metin kontrolü
            else:
                column_types[column] = 'text'
        
        return column_types
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Veriyi temizle ve hataları raporla"""
        cleaned_df = df.copy()
        errors = {}
        
        # Boş değerleri temizle
        cleaned_df = cleaned_df.replace(['', 'nan', 'NaN', 'NULL', 'null'], np.nan)
        
        # Her sütun için temizleme işlemleri
        for column in cleaned_df.columns:
            column_errors = []
            
            # Sayısal sütunlar için
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                # Aykırı değerleri tespit et
                Q1 = cleaned_df[column].quantile(0.25)
                Q3 = cleaned_df[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers = cleaned_df[(cleaned_df[column] < (Q1 - 1.5 * IQR)) | 
                                    (cleaned_df[column] > (Q3 + 1.5 * IQR))][column]
                if not outliers.empty:
                    column_errors.append(f"Aykırı değerler tespit edildi: {outliers.tolist()}")
            
            # Tarih sütunları için
            elif pd.api.types.is_datetime64_any_dtype(cleaned_df[column]):
                # Geçersiz tarihleri kontrol et
                invalid_dates = cleaned_df[pd.to_datetime(cleaned_df[column], errors='coerce').isna()][column]
                if not invalid_dates.empty:
                    column_errors.append(f"Geçersiz tarihler: {invalid_dates.tolist()}")
            
            # Metin sütunları için
            else:
                # Özel karakterleri temizle
                cleaned_df[column] = cleaned_df[column].astype(str).apply(
                    lambda x: re.sub(r'[^\w\s-]', '', x) if pd.notna(x) else x
                )
                
                # Boşlukları temizle
                cleaned_df[column] = cleaned_df[column].astype(str).str.strip()
            
            if column_errors:
                errors[column] = column_errors
        
        return cleaned_df, errors
    
    def validate_data(self, df: pd.DataFrame, validation_rules: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """Veriyi doğrula ve raporla"""
        validation_results = {
            'is_valid': True,
            'errors': {},
            'warnings': {},
            'statistics': {}
        }
        
        # Sütun tiplerini tespit et
        column_types = self.detect_column_types(df)
        
        # Her sütun için doğrulama
        for column in df.columns:
            column_errors = []
            column_warnings = []
            
            # Boş değer kontrolü
            null_count = df[column].isna().sum()
            if null_count > 0:
                column_warnings.append(f"{null_count} boş değer bulundu")
            
            # Özel doğrulama kuralları
            if validation_rules and column in validation_rules:
                for rule in validation_rules[column]:
                    if rule in self.validation_rules:
                        invalid_values = df[~self.validation_rules[rule](df[column])][column]
                        if not invalid_values.empty:
                            column_errors.append(f"Kural ihlali ({rule}): {invalid_values.tolist()}")
            
            # İstatistikler
            if pd.api.types.is_numeric_dtype(df[column]):
                validation_results['statistics'][column] = {
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max()
                }
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                validation_results['statistics'][column] = {
                    'min_date': df[column].min(),
                    'max_date': df[column].max()
                }
            else:
                validation_results['statistics'][column] = {
                    'unique_values': df[column].nunique(),
                    'most_common': df[column].value_counts().head(3).to_dict()
                }
            
            if column_errors:
                validation_results['errors'][column] = column_errors
                validation_results['is_valid'] = False
            
            if column_warnings:
                validation_results['warnings'][column] = column_warnings
        
        return validation_results
    
    def generate_quality_report(self, df: pd.DataFrame, validation_results: Dict[str, Any]) -> str:
        """Veri kalite raporu oluştur"""
        report = []
        report.append("=== VERİ KALİTE RAPORU ===")
        report.append(f"Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Toplam Satır: {len(df)}")
        report.append(f"Toplam Sütun: {len(df.columns)}")
        report.append("\n=== DOĞRULAMA SONUÇLARI ===")
        
        if validation_results['is_valid']:
            report.append("✓ Veri doğrulama başarılı")
        else:
            report.append("✗ Veri doğrulama başarısız")
        
        if validation_results['errors']:
            report.append("\nHatalar:")
            for column, errors in validation_results['errors'].items():
                report.append(f"\n{column}:")
                for error in errors:
                    report.append(f"  - {error}")
        
        if validation_results['warnings']:
            report.append("\nUyarılar:")
            for column, warnings in validation_results['warnings'].items():
                report.append(f"\n{column}:")
                for warning in warnings:
                    report.append(f"  - {warning}")
        
        if validation_results['statistics']:
            report.append("\nİstatistikler:")
            for column, stats in validation_results['statistics'].items():
                report.append(f"\n{column}:")
                for stat_name, stat_value in stats.items():
                    report.append(f"  - {stat_name}: {stat_value}")
        
        return "\n".join(report) 