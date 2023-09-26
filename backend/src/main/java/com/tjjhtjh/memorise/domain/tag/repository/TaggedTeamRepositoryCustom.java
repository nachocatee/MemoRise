package com.tjjhtjh.memorise.domain.tag.repository;

import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedTeam;

import java.util.List;

public interface TaggedTeamRepositoryCustom {
    List<TaggedTeam> findByTaggedTeamWithMemoSeq(Long memoId);
}
