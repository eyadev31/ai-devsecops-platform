import 'package:flutter_test/flutter_test.dart';
import 'package:portfolio_mind_mobile/main.dart';

void main() {
  testWidgets('PortfolioMindApp builds', (WidgetTester tester) async {
    await tester.pumpWidget(const PortfolioMindApp());
    expect(find.text('Portfolio Mind'), findsWidgets);
  });
}

