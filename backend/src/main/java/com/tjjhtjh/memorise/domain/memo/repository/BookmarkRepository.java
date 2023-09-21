package com.tjjhtjh.memorise.domain.memo.repository;

import com.tjjhtjh.memorise.domain.memo.repository.entity.Bookmark;
import org.springframework.data.jpa.repository.JpaRepository;

public interface BookmarkRepository extends JpaRepository<Bookmark,Long> , BookmarkRepositoryCustom{
}
