import 'package:flutter/material.dart';

class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: ListView(
        children: const [
          Text(
            'Profil investisseur',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 16),
          Card(
            child: ListTile(
              leading: Icon(Icons.person),
              title: Text('Nom'),
              subtitle: Text('Eya Khalfallah'),
            ),
          ),
          Card(
            child: ListTile(
              leading: Icon(Icons.trending_up),
              title: Text('Tolérance au risque'),
              subtitle: Text('Modérée'),
            ),
          ),
          Card(
            child: ListTile(
              leading: Icon(Icons.schedule),
              title: Text('Horizon d’investissement'),
              subtitle: Text('Long terme'),
            ),
          ),
          Card(
            child: ListTile(
              leading: Icon(Icons.flag),
              title: Text('Objectif'),
              subtitle: Text('Croissance et diversification'),
            ),
          ),
        ],
      ),
    );
  }
}
