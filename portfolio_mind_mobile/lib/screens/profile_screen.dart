import 'package:flutter/material.dart';
import '../models/user_profile.dart';
import '../services/api_service.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final ApiService _apiService = ApiService();
  late Future<UserProfile> _futureProfile;

  @override
  void initState() {
    super.initState();
    _futureProfile = _apiService.getUserProfile(1);
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<UserProfile>(
      future: _futureProfile,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }

        if (snapshot.hasError) {
          return Center(child: Text('Erreur: ${snapshot.error}'));
        }

        if (!snapshot.hasData) {
          return const Center(child: Text('Aucune donnée'));
        }

        final profile = snapshot.data!;

        return ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Text(
              'Profil investisseur',
              style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),
            Card(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
              child: ListTile(
                leading: const Icon(Icons.person),
                title: const Text('Nom'),
                subtitle: Text(profile.name),
              ),
            ),
            Card(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
              child: ListTile(
                leading: const Icon(Icons.trending_up),
                title: const Text('Tolérance au risque'),
                subtitle: Text(profile.riskLevel),
              ),
            ),
            Card(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
              child: ListTile(
                leading: const Icon(Icons.schedule),
                title: const Text('Horizon d’investissement'),
                subtitle: Text(profile.horizon),
              ),
            ),
            Card(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
              child: ListTile(
                leading: const Icon(Icons.flag),
                title: const Text('Objectif'),
                subtitle: Text(profile.objective),
              ),
            ),
          ],
        );
      },
    );
  }
}
