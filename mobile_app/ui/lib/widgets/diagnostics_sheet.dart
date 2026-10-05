import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../engine/navigation_engine.dart';
import '../utils/theme.dart';

/// Live engine internals and test-drive controls, shown as a bottom sheet.
class DiagnosticsSheet extends StatelessWidget {
  const DiagnosticsSheet({
    super.key,
    required this.state,
    required this.simulate,
    required this.logging,
    required this.logPath,
    required this.useCompass,
    required this.follow,
    required this.mapError,
    required this.onSimulate,
    required this.onLogging,
    required this.onCompass,
    required this.onFollow,
  });

  final ValueListenable<NavState?> state;
  final bool simulate;
  final bool logging;
  final String? logPath;
  final bool useCompass;
  final bool follow;
  final String? mapError;
  final ValueChanged<bool> onSimulate;
  final ValueChanged<bool> onLogging;
  final ValueChanged<bool> onCompass;
  final ValueChanged<bool> onFollow;

  static Future<void> show(BuildContext context, DiagnosticsSheet sheet) => showModalBottomSheet<void>(
        context: context,
        backgroundColor: Colors.transparent,
        isScrollControlled: true,
        builder: (_) => sheet,
      );

  @override
  Widget build(BuildContext context) {
    return _SheetBody(sheet: this);
  }
}

class _SheetBody extends StatefulWidget {
  const _SheetBody({required this.sheet});
  final DiagnosticsSheet sheet;

  @override
  State<_SheetBody> createState() => _SheetBodyState();
}

class _SheetBodyState extends State<_SheetBody> {
  late bool _simulate = widget.sheet.simulate;
  late bool _logging = widget.sheet.logging;
  late bool _compass = widget.sheet.useCompass;
  late bool _follow = widget.sheet.follow;

  @override
  Widget build(BuildContext context) {
    final sheet = widget.sheet;
    return DraggableScrollableSheet(
      initialChildSize: 0.62,
      minChildSize: 0.35,
      maxChildSize: 0.92,
      expand: false,
      builder: (context, controller) => Container(
        decoration: const BoxDecoration(
          color: Color(0xFF0B0B0C),
          borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        ),
        child: ValueListenableBuilder<NavState?>(
          valueListenable: sheet.state,
          builder: (context, s, _) {
            final rows = <(String, String)>[
              ('Mode', s?.mode.name ?? '—'),
              ('GNSS health', s?.gnssHealth.name ?? '—'),
              ('Fix age / accuracy', s == null ? '—' : '${_fmt(s.fixAgeS, 1)} s / ${_fmt(s.fixAccuracyM, 0)} m'),
              ('Fixes accepted / rejected', s == null ? '—' : '${s.acceptedFixes} / ${s.rejectedFixes}'),
              ('Position 2σ', s == null ? '—' : '${_fmt(s.uncertaintyM, 1)} m'),
              ('Heading σ', s == null ? '—' : '${_fmt(s.headingSigmaDeg, 1)}°'),
              ('Speed σ', s == null ? '—' : '${_fmt(s.speedSigmaMs, 2)} m/s'),
              ('Gyro bias', s == null ? '—' : '${_fmt(s.gyroBiasDps, 3)} °/s'),
              ('Accel bias', s == null ? '—' : '${_fmt(s.accelBiasMs2, 3)} m/s²'),
              ('Alignment', s?.alignment.name ?? '—'),
              ('Mount (pitch/roll/yaw)', s?.mount?.toString() ?? 'not yet'),
              (
                'Calibration',
                s?.calibration == null
                    ? '—'
                    : 'fwd r=${_fmt(s!.calibration!.forwardCorr, 2)} · yaw r=${_fmt(s.calibration!.yawCorr, 2)} · '
                        'compass R=${_fmt(s.calibration!.compassR, 2)} · ${_fmt(s.calibration!.windowSeconds, 0)} s'
              ),
              ('Vibration RMS / shocks', s == null ? '—' : '${_fmt(s.vibrationRms, 2)} m/s² / ${s.shockCount}'),
              ('Road graph', s == null ? '—' : '${s.roadSegments} segments${sheet.mapError != null ? ' · ${sheet.mapError}' : ''}'),
              ('Blackout', s == null || !s.isDeadReckoning ? '—' : '${_fmt(s.blackoutDistanceM, 0)} m · ${_fmt(s.blackoutSeconds, 1)} s · ${s.mapAided ? 'on road' : 'free'}'),
              (
                'Last recovery',
                s?.lastRecovery == null
                    ? '—'
                    : '${_fmt(s!.lastRecovery!.distanceM, 0)} m, ${_fmt(s.lastRecovery!.errorM, 1)} m off (${_fmt(s.lastRecovery!.driftPct, 1)} %)'
              ),
              ('Log file', _logging ? (sheet.logPath ?? 'starting…') : 'off'),
            ];

            return ListView(
              controller: controller,
              padding: const EdgeInsets.fromLTRB(20, 12, 20, 32),
              children: [
                Center(
                  child: Container(
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(color: Colors.white24, borderRadius: BorderRadius.circular(2)),
                  ),
                ),
                const SizedBox(height: 14),
                Text('ENGINE DIAGNOSTICS', style: HudTheme.hudLabel()),
                const SizedBox(height: 10),
                _toggle('Simulate GNSS loss', _simulate, (v) {
                  setState(() => _simulate = v);
                  sheet.onSimulate(v);
                }),
                _toggle('Record drive log (CSV)', _logging, (v) {
                  setState(() => _logging = v);
                  sheet.onLogging(v);
                }),
                _toggle('Compass pull in outages', _compass, (v) {
                  setState(() => _compass = v);
                  sheet.onCompass(v);
                }),
                _toggle('Follow vehicle', _follow, (v) {
                  setState(() => _follow = v);
                  sheet.onFollow(v);
                }),
                const SizedBox(height: 8),
                for (final (k, v) in rows)
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 5),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        SizedBox(width: 150, child: Text(k, style: HudTheme.hudLabel())),
                        Expanded(
                          child: Text(v, style: HudTheme.hudSmall(HudTheme.textPrimary).copyWith(fontSize: 13)),
                        ),
                      ],
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }

  static Widget _toggle(String label, bool value, ValueChanged<bool> onChanged) => SwitchListTile(
        contentPadding: EdgeInsets.zero,
        dense: true,
        title: Text(label, style: HudTheme.hudSmall(HudTheme.textPrimary)),
        value: value,
        activeThumbColor: HudTheme.drAccent,
        onChanged: onChanged,
      );

  static String _fmt(double v, int d) => v.isFinite ? v.toStringAsFixed(d) : '—';
}
