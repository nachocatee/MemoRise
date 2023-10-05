package com.tjjhtjh.memorise.domain.tag.repository;

import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedTeam;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TaggedTeamRepository extends JpaRepository<TaggedTeam, Long>, TaggedTeamRepositoryCustom {
}
