import 'dart:async';
import 'package:geolocator/geolocator.dart';

/// Wraps [Geolocator] and exposes a continuous GPS position stream.
class LocationService {
  StreamSubscription<Position>? _subscription;
  final StreamController<Position> _controller =
      StreamController<Position>.broadcast();

  /// Broadcast stream of GPS positions.
  Stream<Position> get positionStream => _controller.stream;

  /// Checks if location services are enabled and permissions are granted.
  /// Returns `true` if ready, `false` otherwise.
  static Future<bool> checkPermissions() async {
    if (!await Geolocator.isLocationServiceEnabled()) return false;
    final perm = await Geolocator.checkPermission();
    return perm == LocationPermission.always ||
        perm == LocationPermission.whileInUse;
  }

  /// Request location permission. Returns true if granted.
  static Future<bool> requestPermission() async {
    var perm = await Geolocator.checkPermission();
    if (perm == LocationPermission.denied) {
      perm = await Geolocator.requestPermission();
    }
    return perm == LocationPermission.always ||
        perm == LocationPermission.whileInUse;
  }

  /// Start listening for position updates.
  void startListening() {
    _subscription?.cancel();

    const settings = LocationSettings(
      accuracy: LocationAccuracy.bestForNavigation,
      distanceFilter: 0, // every update
    );

    _subscription =
        Geolocator.getPositionStream(locationSettings: settings).listen(
      (position) {
        if (!_controller.isClosed) {
          _controller.add(position);
        }
      },
      onError: (error) {
        // Stream errors (e.g. GPS turned off) – silently ignore,
        // dashboard will detect via accuracy going stale.
      },
    );
  }

  /// Stop listening.
  void stopListening() {
    _subscription?.cancel();
    _subscription = null;
  }

  /// Clean up resources.
  void dispose() {
    stopListening();
    _controller.close();
  }
}
