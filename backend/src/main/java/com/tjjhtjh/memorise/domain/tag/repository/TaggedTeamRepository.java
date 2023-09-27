package com.tjjhtjh.memorise.domain.tag.repository;

import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedTeam;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface TaggedTeamRepository extends JpaRepository<TaggedTeam, Long>, TaggedTeamRepositoryCustom {
}
