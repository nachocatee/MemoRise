package com.tjjhtjh.memorise.domain.team.repository;

import com.tjjhtjh.memorise.domain.team.repository.entity.TeamUser;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface TeamUserRepository extends JpaRepository<TeamUser,Long>, TeamUserSupprtRepository {
}
