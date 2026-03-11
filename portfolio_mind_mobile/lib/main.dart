import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const PortfolioMindApp());
}

class PortfolioMindApp extends StatelessWidget {
  const PortfolioMindApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Portfolio Mind',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}
