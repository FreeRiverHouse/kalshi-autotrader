#!/usr/bin/env python3
"""
Kalshi Weekly Trading Report - PDF Generator
Generates a comprehensive weekly trading report in PDF format.
Runs every Sunday via cron.
"""

import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

TRADES_FILE = Path(__file__).parent / "kalshi-trades.jsonl"
STOP_LOSS_LOG = Path(__file__).parent / "kalshi-stop-loss.log"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "reports"

def parse_timestamp(ts):
    """Parse ISO timestamp."""
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))

def get_pst_date(dt):
    """Convert datetime to PST date string."""
    pst_offset = timedelta(hours=-8)
    pst_time = dt + pst_offset
    return pst_time.strftime('%Y-%m-%d')

def get_pst_now():
    """Get current datetime in PST."""
    now_utc = datetime.now(timezone.utc)
    pst_offset = timedelta(hours=-8)
    return now_utc + pst_offset

def analyze_stop_losses_week(week_start_str: str) -> dict:
    """
    Analyze stop-losses triggered in the past week.
    Returns stats: count, total_loss, avg_loss_pct, by_day breakdown.
    """
    if not STOP_LOSS_LOG.exists():
        return {"count": 0, "total_loss_cents": 0, "avg_loss_pct": 0, "by_day": {}}
    
    week_start = datetime.fromisoformat(week_start_str).replace(tzinfo=timezone.utc)
    stops_by_day = defaultdict(lambda: {"count": 0, "loss_cents": 0})
    all_stops = []
    
    with open(STOP_LOSS_LOG) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") != "stop_loss":
                    continue
                ts = entry.get("timestamp", "")
                if not ts:
                    continue
                dt = parse_timestamp(ts)
                if dt < week_start:
                    continue
                
                entry_price = entry.get("entry_price", 0) or entry.get("entry_price_cents", 0)
                exit_price = entry.get("exit_price", 0) or entry.get("exit_price_cents", 0)
                contracts = entry.get("contracts", 1)
                loss_pct = entry.get("loss_pct", 0)
                
                loss_cents = (entry_price - exit_price) * contracts
                date_str = get_pst_date(dt)
                
                stops_by_day[date_str]["count"] += 1
                stops_by_day[date_str]["loss_cents"] += loss_cents
                all_stops.append({"loss_cents": loss_cents, "loss_pct": loss_pct})
            except:
                pass
    
    if not all_stops:
        return {"count": 0, "total_loss_cents": 0, "avg_loss_pct": 0, "by_day": {}}
    
    total_loss = sum(s["loss_cents"] for s in all_stops)
    avg_loss_pct = sum(s["loss_pct"] for s in all_stops) / len(all_stops)
    
    return {
        "count": len(all_stops),
        "total_loss_cents": total_loss,
        "avg_loss_pct": avg_loss_pct,
        "by_day": dict(stops_by_day)
    }

def analyze_week():
    """Analyze last 7 days of trades."""
    now_pst = get_pst_now()
    week_ago = now_pst - timedelta(days=7)
    
    trades = []
    daily_stats = defaultdict(lambda: {'trades': 0, 'won': 0, 'lost': 0, 'pending': 0, 'pnl': 0, 'wagered': 0})
    
    # Momentum correlation tracking
    momentum_stats = {
        'aligned': {'trades': 0, 'won': 0, 'lost': 0, 'pnl': 0},
        'not_aligned': {'trades': 0, 'won': 0, 'lost': 0, 'pnl': 0},
        'unknown': {'trades': 0, 'won': 0, 'lost': 0, 'pnl': 0},
    }
    
    # Regime breakdown tracking
    regime_stats = defaultdict(lambda: {'trades': 0, 'won': 0, 'lost': 0, 'pnl': 0})
    
    with open(TRADES_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get('type') != 'trade' or entry.get('order_status') != 'executed':
                    continue
                    
                trade_time = parse_timestamp(entry['timestamp'])
                if trade_time < week_ago.replace(tzinfo=timezone.utc):
                    continue
                    
                trade_date = get_pst_date(trade_time)
                trades.append(entry)
                
                cost = entry.get('cost_cents', 0)
                contracts = entry.get('contracts', 1)
                result = entry.get('result_status', 'pending')
                
                daily_stats[trade_date]['trades'] += 1
                daily_stats[trade_date]['wagered'] += cost
                
                # Track momentum alignment
                mom_aligned = entry.get('momentum_aligned')
                if mom_aligned is True:
                    mom_key = 'aligned'
                elif mom_aligned is False:
                    mom_key = 'not_aligned'
                else:
                    mom_key = 'unknown'
                
                momentum_stats[mom_key]['trades'] += 1
                
                # Track regime
                regime = entry.get('regime', 'unknown')
                regime_stats[regime]['trades'] += 1
                
                if result == 'won':
                    daily_stats[trade_date]['won'] += 1
                    daily_stats[trade_date]['pnl'] += (contracts * 100) - cost
                    entry['_pnl'] = (contracts * 100) - cost
                    momentum_stats[mom_key]['won'] += 1
                    momentum_stats[mom_key]['pnl'] += (contracts * 100) - cost
                    regime_stats[regime]['won'] += 1
                    regime_stats[regime]['pnl'] += (contracts * 100) - cost
                elif result == 'lost':
                    daily_stats[trade_date]['lost'] += 1
                    daily_stats[trade_date]['pnl'] -= cost
                    entry['_pnl'] = -cost
                    momentum_stats[mom_key]['lost'] += 1
                    momentum_stats[mom_key]['pnl'] -= cost
                    regime_stats[regime]['lost'] += 1
                    regime_stats[regime]['pnl'] -= cost
                else:
                    daily_stats[trade_date]['pending'] += 1
                    entry['_pnl'] = 0
                    
            except (json.JSONDecodeError, KeyError):
                continue
    
    # Sort trades by PnL for best/worst
    settled_trades = [t for t in trades if t.get('result_status') in ('won', 'lost')]
    settled_trades.sort(key=lambda x: x.get('_pnl', 0), reverse=True)
    
    best_trade = settled_trades[0] if settled_trades else None
    worst_trade = settled_trades[-1] if settled_trades else None
    
    # Calculate totals
    total_trades = len(trades)
    total_won = sum(d['won'] for d in daily_stats.values())
    total_lost = sum(d['lost'] for d in daily_stats.values())
    total_pending = sum(d['pending'] for d in daily_stats.values())
    total_pnl = sum(d['pnl'] for d in daily_stats.values())
    total_wagered = sum(d['wagered'] for d in daily_stats.values())
    
    settled = total_won + total_lost
    win_rate = (total_won / settled * 100) if settled > 0 else 0
    
    # Calculate momentum win rates
    for key in momentum_stats:
        s = momentum_stats[key]
        settled = s['won'] + s['lost']
        s['win_rate'] = (s['won'] / settled * 100) if settled > 0 else None
    
    # Calculate regime win rates
    for key in regime_stats:
        s = regime_stats[key]
        settled = s['won'] + s['lost']
        s['win_rate'] = (s['won'] / settled * 100) if settled > 0 else None
    
    week_start_str = (now_pst - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # Get stop-loss stats for the week
    stop_loss_stats = analyze_stop_losses_week(week_start_str)
    
    return {
        'week_start': week_start_str,
        'week_end': now_pst.strftime('%Y-%m-%d'),
        'total_trades': total_trades,
        'total_won': total_won,
        'total_lost': total_lost,
        'total_pending': total_pending,
        'total_pnl': total_pnl,
        'total_wagered': total_wagered,
        'win_rate': win_rate,
        'daily_stats': dict(daily_stats),
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'momentum_stats': momentum_stats,
        'regime_stats': dict(regime_stats),
        'stop_loss_stats': stop_loss_stats,
    }

def generate_pdf(stats):
    """Generate PDF report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font('Helvetica', 'B', 24)
    pdf.cell(0, 15, 'Kalshi Trading Report', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, f"Week: {stats['week_start']} to {stats['week_end']}", align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    
    # Summary Box
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Weekly Summary', new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    
    pdf.set_font('Helvetica', '', 11)
    pdf.ln(3)
    
    # Stats grid
    col_width = 95
    pdf.cell(col_width, 7, f"Total Trades: {stats['total_trades']}", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(col_width, 7, f"Total Wagered: ${stats['total_wagered']/100:.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.cell(col_width, 7, f"Trades Won: {stats['total_won']}", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(col_width, 7, f"Trades Lost: {stats['total_lost']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.cell(col_width, 7, f"Pending: {stats['total_pending']}", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pnl_color = (0, 128, 0) if stats['total_pnl'] >= 0 else (200, 0, 0)
    pdf.cell(col_width, 7, f"Win Rate: {stats['win_rate']:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(3)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(*pnl_color)
    pdf.cell(0, 10, f"Weekly PnL: ${stats['total_pnl']/100:+.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    # Daily Breakdown
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Daily Breakdown', new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(3)
    
    # Table header
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(35, 8, 'Date', border=1, fill=True)
    pdf.cell(25, 8, 'Trades', border=1, align='C', fill=True)
    pdf.cell(25, 8, 'Won', border=1, align='C', fill=True)
    pdf.cell(25, 8, 'Lost', border=1, align='C', fill=True)
    pdf.cell(35, 8, 'Wagered', border=1, align='R', fill=True)
    pdf.cell(35, 8, 'PnL', border=1, align='R', fill=True)
    pdf.ln()
    
    # Table rows
    pdf.set_font('Helvetica', '', 10)
    for date in sorted(stats['daily_stats'].keys(), reverse=True):
        day = stats['daily_stats'][date]
        pdf.cell(35, 7, date, border=1)
        pdf.cell(25, 7, str(day['trades']), border=1, align='C')
        pdf.cell(25, 7, str(day['won']), border=1, align='C')
        pdf.cell(25, 7, str(day['lost']), border=1, align='C')
        pdf.cell(35, 7, f"${day['wagered']/100:.2f}", border=1, align='R')
        
        pnl = day['pnl']
        if pnl >= 0:
            pdf.set_text_color(0, 128, 0)
        else:
            pdf.set_text_color(200, 0, 0)
        pdf.cell(35, 7, f"${pnl/100:+.2f}", border=1, align='R')
        pdf.set_text_color(0, 0, 0)
        pdf.ln()
    
    pdf.ln(8)
    
    # Best/Worst Trades
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Notable Trades', new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(3)
    
    pdf.set_font('Helvetica', '', 10)
    
    if stats['best_trade']:
        t = stats['best_trade']
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 7, f"Best Trade: {t.get('ticker', 'N/A')} @ {t.get('avg_price', 0)}c "
                       f"({t.get('side', 'N/A')}) -> ${t.get('_pnl', 0)/100:+.2f}", 
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    if stats['worst_trade']:
        t = stats['worst_trade']
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 7, f"Worst Trade: {t.get('ticker', 'N/A')} @ {t.get('avg_price', 0)}c "
                       f"({t.get('side', 'N/A')}) -> ${t.get('_pnl', 0)/100:+.2f}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Momentum Analysis Section
    mom_stats = stats.get('momentum_stats', {})
    if any(s.get('trades', 0) > 0 for s in mom_stats.values()):
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Momentum Correlation Analysis', new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        pdf.ln(3)
        
        pdf.set_font('Helvetica', '', 10)
        
        aligned = mom_stats.get('aligned', {})
        not_aligned = mom_stats.get('not_aligned', {})
        
        # Show aligned vs not aligned comparison
        if aligned.get('trades', 0) > 0:
            wr = aligned.get('win_rate')
            wr_str = f"{wr:.1f}%" if wr is not None else "N/A"
            pdf.cell(0, 7, f"Momentum Aligned Trades: {aligned['trades']} | "
                          f"Win Rate: {wr_str} | PnL: ${aligned['pnl']/100:+.2f}",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        if not_aligned.get('trades', 0) > 0:
            wr = not_aligned.get('win_rate')
            wr_str = f"{wr:.1f}%" if wr is not None else "N/A"
            pdf.cell(0, 7, f"Non-Aligned Trades: {not_aligned['trades']} | "
                          f"Win Rate: {wr_str} | PnL: ${not_aligned['pnl']/100:+.2f}",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Show insight
        aligned_wr = aligned.get('win_rate') or 0
        not_aligned_wr = not_aligned.get('win_rate') or 0
        if aligned.get('trades', 0) > 0 and not_aligned.get('trades', 0) > 0:
            diff = aligned_wr - not_aligned_wr
            if diff > 5:
                insight = f"Momentum alignment improves win rate by {diff:.1f}pp"
                pdf.set_text_color(0, 128, 0)
            elif diff < -5:
                insight = f"Momentum alignment hurts win rate by {abs(diff):.1f}pp (investigate!)"
                pdf.set_text_color(200, 0, 0)
            else:
                insight = "No significant momentum correlation detected"
                pdf.set_text_color(128, 128, 128)
            pdf.set_font('Helvetica', 'I', 10)
            pdf.cell(0, 7, f"→ {insight}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
        
        pdf.ln(5)
    
    # Regime Breakdown Section
    regime_stats = stats.get('regime_stats', {})
    if any(s.get('trades', 0) > 0 for s in regime_stats.values()):
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Performance by Market Regime', new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        pdf.ln(3)
        
        pdf.set_font('Helvetica', '', 10)
        
        # Sort by trades descending
        sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1].get('trades', 0), reverse=True)
        
        for regime, rs in sorted_regimes:
            if rs.get('trades', 0) == 0:
                continue
            wr = rs.get('win_rate')
            wr_str = f"{wr:.1f}%" if wr is not None else "N/A"
            regime_label = regime.replace('_', ' ').title()
            pdf.cell(0, 7, f"{regime_label}: {rs['trades']} trades | "
                          f"Win Rate: {wr_str} | PnL: ${rs['pnl']/100:+.2f}",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.ln(8)
    
    # Stop-Loss Summary Section
    sl_stats = stats.get('stop_loss_stats', {})
    if sl_stats.get('count', 0) > 0:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Stop-Loss Summary', new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        pdf.ln(3)
        
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 7, f"Total Stop-Losses: {sl_stats['count']}", 
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 7, f"Total Loss at Exit: ${sl_stats['total_loss_cents']/100:.2f}", 
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, f"Average Loss %: {sl_stats['avg_loss_pct']:.1f}%", 
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Show daily breakdown if multiple days
        by_day = sl_stats.get('by_day', {})
        if len(by_day) > 1:
            pdf.ln(3)
            pdf.set_font('Helvetica', 'I', 9)
            for date in sorted(by_day.keys()):
                day_stats = by_day[date]
                pdf.cell(0, 6, f"  {date}: {day_stats['count']} stop-loss(es), "
                              f"${day_stats['loss_cents']/100:.2f} loss",
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.ln(5)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 6, "Note: Stop-losses trigger when position value drops 50% from entry.",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(8)
    
    # Footer
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Autotrader v2 | Black-Scholes Model",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Save
    filename = f"kalshi-weekly-{stats['week_end']}.pdf"
    output_path = OUTPUT_DIR / filename
    pdf.output(str(output_path))
    
    print(f"Report saved to: {output_path}")
    return output_path

def main():
    stats = analyze_week()
    
    # Print summary to stdout
    print(f"\n📊 Weekly Report: {stats['week_start']} → {stats['week_end']}")
    print(f"   Trades: {stats['total_trades']} | Won: {stats['total_won']} | Lost: {stats['total_lost']}")
    print(f"   Win Rate: {stats['win_rate']:.1f}% | PnL: ${stats['total_pnl']/100:+.2f}")
    
    # Generate PDF
    pdf_path = generate_pdf(stats)
    
    # Write alert file if there were trades
    if stats['total_trades'] > 0:
        alert_file = Path(__file__).parent / "kalshi-weekly-report.alert"
        with open(alert_file, 'w') as f:
            f.write(f"📊 **Weekly Trading Report Ready**\n\n")
            f.write(f"Week: {stats['week_start']} → {stats['week_end']}\n")
            f.write(f"Trades: {stats['total_trades']}\n")
            f.write(f"Win Rate: {stats['win_rate']:.1f}%\n")
            f.write(f"PnL: ${stats['total_pnl']/100:+.2f}\n\n")
            f.write(f"PDF: {pdf_path}")
        print(f"Alert file written: {alert_file}")
    
    return pdf_path

if __name__ == "__main__":
    main()
