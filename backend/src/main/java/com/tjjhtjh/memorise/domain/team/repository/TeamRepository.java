package com.tjjhtjh.memorise.domain.team.repository;

import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TeamRepository extends JpaRepository<Team,Long>, TeamSupportRepository {
}
