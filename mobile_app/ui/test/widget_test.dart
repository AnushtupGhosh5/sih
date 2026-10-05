import 'package:flutter_test/flutter_test.dart';
import 'package:sih_navigation/main.dart';

void main() {
  testWidgets('App builds without error', (WidgetTester tester) async {
    await tester.pumpWidget(const SihNavigationApp());
    // The permission screen should render.
    expect(find.text('SIH NAVIGATION'), findsOneWidget);
  });
}
