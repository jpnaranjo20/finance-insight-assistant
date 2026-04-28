from fpdf import FPDF
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import smtplib
import os
from datetime import datetime
from email_validator import validate_email, EmailNotValidError
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
load_dotenv()

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import random

class EmailSchema(BaseModel):
    """Schema para validar email."""
    email: EmailStr
    symbol: str

def create_stock_report(symbol: str, price_data: dict, metrics: dict, analysis: dict) -> str:
    """
    Genera un reporte PDF más profesional:
      1. Sección de Visión General con el análisis detallado.
      2. Gráficos insertados en el PDF (histórico de precios y radar de métricas).
      3. Sección de Conclusiones o recomendaciones.
    """

    # 1) Preparamos los gráficos Plotly y los guardamos como PNG
    # Gráfico 1: Histórico de precios simulado
    fig1 = go.Figure()
    base_price = price_data["current_price"]
    dates = pd.date_range(end=datetime.now(), periods=30)
    prices = [base_price * (1 + random.uniform(-0.1, 0.1)) for _ in range(30)]
    fig1.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers', name=f'Histórico {symbol}'))
    fig1.update_layout(
        title=f"Histórico de precios - {symbol}",
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        template="simple_white"
    )
    chart1_name = f"temp_price_{symbol}.png"
    pio.write_image(fig1, chart1_name, width=700, height=400)  # Exportar a PNG

    # Gráfico 2: Radar de métricas (ejemplo)
    # Ej: tomamos P/E, Beta, EPS, Debt/Equity, Dividend yield
    # Convertimos el % de dividend yield a float
    try:
        dividend_float = float(metrics["dividend_yield"].strip('%'))
    except:
        dividend_float = 0.0
    
    r_values = [
        metrics["pe_ratio"],
        metrics["beta"],
        metrics["eps"],
        metrics["debt_to_equity"],
        dividend_float
    ]
    theta_values = ["P/E ratio", "Beta", "EPS", "Debt/Equity", "Dividend Yield"]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=r_values,
        theta=theta_values,
        fill='toself',
        name=f'Métricas clave {symbol}'
    ))
    fig2.update_layout(
        title=f"Métricas clave - {symbol}",
        polar=dict(radialaxis=dict(visible=True, range=[0, max(r_values)+5])),
        template="simple_white"
    )
    chart2_name = f"temp_metrics_{symbol}.png"
    pio.write_image(fig2, chart2_name, width=600, height=500)

    #n2) Construimos el PDF con FPDF

    pdf = FPDF()
    pdf.add_page()

    # Título principal
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'Reporte Financiero - {symbol}', ln=True, align='C')
    pdf.ln(8)

    # VISIÓN GENERAL (Análisis)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Visión General', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 8, analysis["analysis"], ln=True)  # Añade el texto del análisis en múltiples líneas
    pdf.ln(5)

    # Agregar primer gráfico: Histórico de Precios
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Gráfico: Histórico de Precios', ln=True)
    # Insertamos la imagen en el PDF (ajusta x, y, w, h según desees)
    pdf.image(chart1_name, x=10, y=None, w=180)
    pdf.ln(80)  # Deja un espacio debajo de la imagen (dependiendo del alto que ocupó)

    # Agregar segundo gráfico: Radar de Métricas
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Gráfico: Métricas Financieras (Radar)', ln=True)
    pdf.image(chart2_name, x=15, y=None, w=160)
    pdf.ln(10)

    # INFORMACIÓN BÁSICA
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Información Básica de la Acción', ln=True)
    pdf.set_font('Arial', '', 11)

    pdf.cell(0, 10, f"Precio Actual: {price_data['current_price']} {price_data['currency']}", ln=True)
    pdf.cell(0, 10, f"Estado: {price_data['status']}", ln=True)
    pdf.cell(0, 10, f"Sector: {price_data['sector']}", ln=True)
    pdf.cell(0, 10, f"Volumen: {price_data.get('volume', 'N/A')}", ln=True)
    pdf.cell(0, 10, f"Cambio %: {price_data.get('change_percent', 'N/A')}%", ln=True)
    pdf.ln(5)

    # MÉTRICAS CLAVE
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Métricas Clave', ln=True)
    pdf.set_font('Arial', '', 11)
    for key, value in metrics.items():
        pdf.cell(0, 10, f'{key}: {value}', ln=True)

    pdf.ln(5)

    # CONCLUSIONES / RECOMENDACIONES
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Conclusiones y Recomendaciones', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 8, (
        f"Recomendación actual: {analysis['recommendation'].upper()}\n"
        f"Nivel de riesgo estimado: {analysis['risk_level'].capitalize()}\n"
        f"Precio objetivo sugerido: ${analysis['target_price']}\n\n"
        "Sugerimos evaluar la diversificación de portafolio y mantenerse atento "
        "a posibles cambios en la política monetaria que puedan impactar el mercado."
    ))
    
    # Guardamos el PDF en un archivo temporal (por ahora en el root del proyecto)
    filename = f'report_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    pdf.output(filename)

    # Eliminamos las imágenes temporales
    if os.path.exists(chart1_name):
        os.remove(chart1_name)
    if os.path.exists(chart2_name):
        os.remove(chart2_name)

    return filename

def send_email(to_email: str, report_file: str, symbol: str) -> str:
    """Envía el reporte por email con formato HTML profesional."""
    from_email = "g4technologies.dev@gmail.com"
    password = os.getenv("EMAIL_PASSWORD")  # hacer load_dotenv() antes

    msg = MIMEMultipart("alternative")
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = f'Reporte Financiero - {symbol}'

    # Contenido del cuerpo en HTML
    body_html = f"""
    <html>
    <head>
      <meta charset="utf-8">
    </head>
    <body style="font-family: Arial, sans-serif; color: #333;">
      <p>Estimado(a) inversionista,</p>

      <p>
        Reciba un cordial saludo de parte del equipo de 
        <strong>Asesoría Financiera Profesional</strong>. 
        Adjunto encontrará el reporte detallado sobre la acción 
        <strong>{symbol}</strong>, elaborado con un análisis profundo de los 
        principales indicadores financieros, tendencias de mercado y proyecciones 
        futuras.
      </p>

      <p>
        En este informe incluimos:
      </p>

      <ul>
        <li><strong>Visión General</strong> con una descripción detallada de la situación actual 
            y las perspectivas de la acción.</li>
        <li><strong>Gráficos</strong> que muestran la evolución histórica de precios y 
            un radar de métricas financieras clave.</li>
        <li><strong>Recomendaciones Estratégicas</strong> basadas en la 
            evaluación de escenarios de riesgo y oportunidades potenciales en el mercado bursátil.</li>
      </ul>

      <p>
        Nuestro objetivo es brindarle información confiable y oportuna para facilitar 
        sus decisiones de inversión. Si requiere asistencia adicional para interpretar 
        los datos presentados o desea un asesoramiento más especializado, no dude en 
        ponerse en contacto con nosotros.
      </p>

      <p>
        Agradecemos su confianza en nuestros servicios de asesoría y esperamos que este 
        reporte sea de gran utilidad para optimizar su estrategia de inversión.
      </p>

      <p>
        <em>Atentamente,</em><br/>
        <strong>El Equipo de Asesoría Financiera Profesional</strong>
      </p>
    </body>
    </html>
    """

    # Creamos la parte HTML del mensaje
    html_part = MIMEText(body_html, 'html')
    msg.attach(html_part)

    # Adjuntamos el archivo PDF
    with open(report_file, 'rb') as f:
        attachment = MIMEApplication(f.read(), _subtype='pdf')
        attachment.add_header('Content-Disposition', 'attachment', filename=report_file)
        msg.attach(attachment)

    # Enviamos el correo
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()
        
        # (opcional) eliminar el archivo temporal del reporte
        os.remove(report_file)
        
        return "Reporte enviado exitosamente"
    except Exception as e:
        return f"Error al enviar email: {str(e)}"
