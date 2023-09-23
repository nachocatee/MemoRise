package com.tjjhtjh.memorise.domain.memo.repository;

import com.querydsl.core.types.Projections;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.memo.repository.entity.AccessType;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoResponse;
import org.springframework.data.jpa.repository.support.QuerydslRepositorySupport;
import org.springframework.stereotype.Repository;

import java.util.List;

import static com.tjjhtjh.memorise.domain.memo.repository.entity.QMemo.memo;

@Repository
public class MemoRepositoryImpl extends QuerydslRepositorySupport implements MemoRepositoryCustom {

    private final JPAQueryFactory queryFactory;

    public MemoRepositoryImpl(JPAQueryFactory jpaQueryFactory){
        super(Memo.class);
        this.queryFactory = jpaQueryFactory;
    }

    @Override
    public List<MemoResponse> itemMemoListofOpen(Long itemSeq, Long userSeq) {
        return queryFactory.select(Projections.fields
                (MemoResponse.class,
                        memo.user.nickname.as("nickname"), memo.updatedAt,memo.content,memo.accessType,memo.file))
                .from(memo)
                .leftJoin(memo.user)
                .where(memo.isDeleted.eq(0).and(memo.item.itemSeq.eq(itemSeq).and(memo.accessType.eq(AccessType.OPEN))))
                .fetch();
    }

    @Override
    public List<MemoResponse> itemMemoListofClosed(Long itemSeq, Long userSeq) {
        return queryFactory.select(Projections.fields
                        (MemoResponse.class,
                                memo.user.nickname.as("nickname"), memo.updatedAt,memo.content,memo.accessType,memo.file))
                .from(memo)
                .leftJoin(memo.user)
                .where(memo.isDeleted.eq(0).and(memo.item.itemSeq.eq(itemSeq).and(memo.accessType.eq(AccessType.CLOSED)).and(memo.user.userSeq.eq(userSeq))))
                .fetch();
    }

    @Override
    public List<MemoResponse> itemMemoListofRestictMe(Long itemSeq, Long userSeq) {
        return queryFactory.select(Projections.fields
                        (MemoResponse.class,
                                memo.user.nickname.as("nickname"), memo.updatedAt,memo.content,memo.accessType,memo.file))
                .from(memo)
                .leftJoin(memo.user)
                .where(memo.isDeleted.eq(0).and(memo.item.itemSeq.eq(itemSeq).and(memo.accessType.eq(AccessType.RESTRICT)).and(memo.user.userSeq.eq(userSeq))))
                .fetch();
    }

}
